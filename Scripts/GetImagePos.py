"""
RTABMapRelocator 是一个用于将RTABMap数据库中的位姿数据转换为Unity兼容格式的工具类。
优化点：引入高质量特征筛选算法（Lowe's Ratio Test），修正置信度爆表问题。
"""

import cv2
import numpy as np
import sqlite3
from scipy.spatial.transform import Rotation as R  # type: ignore

class RTABMapRelocator:
    """
    RTABMap重定位器类，用于从RTABMap数据库中查找图像位置并将位姿转换为Unity兼容格式。
    """
    
    def __init__(self, db_path: str) -> None:
        """
        初始化RTABMap重定位器。
        """
        self.m_dbPath = db_path
        # 增加特征点数量以提高召回率
        self.m_orb = cv2.ORB_create(nfeatures=2000)  # type: ignore
        # 注意：使用 KNN 匹配时，crossCheck 必须为 False
        self.m_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # type: ignore
        self.m_targetSize: tuple[int, int] | None = None

    def _get_db_image_format(self, cursor) -> tuple[int, int] | None:
        """
        获取数据库中图像的尺寸格式。
        """
        cursor.execute("SELECT image FROM Data LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            nparr = np.frombuffer(row[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            return img.shape[::-1]  # type: ignore (width, height)
        return None

    def find_location(self, query_image_path: str) -> dict[str, any] | str: # type: ignore
        """
        在RTABMap数据库中查找与查询图像最匹配的位置。
        """
        conn = sqlite3.connect(self.m_dbPath)
        cursor = conn.cursor()

        self.m_targetSize = self._get_db_image_format(cursor)
        raw_query = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if raw_query is None:
            conn.close()
            return "Error: Image not found"

        # 尺寸对齐与直方图均衡化（增强对比度有利于特征提取）
        if self.m_targetSize:
            raw_query = cv2.resize(raw_query, self.m_targetSize, interpolation=cv2.INTER_AREA)
        query_img = cv2.equalizeHist(raw_query)

        # 提取查询图特征
        kp_q, des_q = self.m_orb.detectAndCompute(query_img, None)
        if des_q is None:
            conn.close()
            return "Error: No features found in query image"
        
        m_totalQueryFeatures = len(kp_q)

        # 获取数据库中所有有效的节点和图像
        cursor.execute(
            "SELECT Node.id, Data.image, Node.pose FROM Node JOIN Data ON Node.id = Data.id"
        )
        rows = cursor.fetchall()

        m_bestNodeId = -1
        m_maxGoodMatches = 0
        m_bestPose = None

        for node_id, img_blob, pose_blob in rows:
            if not img_blob or not pose_blob:
                continue

            # 解码数据库图像
            nparr = np.frombuffer(img_blob, np.uint8)
            map_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            map_img = cv2.equalizeHist(map_img) # type: ignore

            # 提取地图帧特征
            kp_m, des_m = self.m_orb.detectAndCompute(map_img, None)
            if des_m is None:
                continue

            # --- 核心优化：高质量匹配筛选 ---
            # 使用 k=2 进行最近邻和次近邻搜索
            matches = self.m_bf.knnMatch(des_q, des_m, k=2)
            
            # Lowe's ratio test: 只有当最近邻明显优于次近邻时，才认为是可靠匹配
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance: # 0.75 为标准过滤系数
                        good_matches.append(m)

            current_good_count = len(good_matches)
            if current_good_count > m_maxGoodMatches:
                m_maxGoodMatches = current_good_count
                m_bestNodeId = node_id
                # 转换 Pose 字节流为矩阵
                m_bestPose = np.frombuffer(pose_blob, dtype=np.float32).reshape(3, 4)

        conn.close()

        # --- 优化后的置信度算法 ---
        # 1. match_ratio: 过滤后的好匹配点占总查询点的比例
        # 2. count_weight: 如果匹配点数太少（比如低于40个），则置信度按比例惩罚
        if m_bestNodeId != -1:
            match_ratio = m_maxGoodMatches / m_totalQueryFeatures
            count_weight = min(m_maxGoodMatches / 40.0, 1.0) # 满40个好匹配点不再惩罚
            
            # 这里的 300 是映射系数，意味着高质量匹配率达到 33% 时得分为 100
            m_confidence = (match_ratio * 300) * count_weight * 100
            m_confidence = min(round(m_confidence, 2), 100.0)

            # 门槛值：经过筛选的 Good Matches 只要超过 10% 置信度通常就有参考价值
            if m_confidence > 10.0:
                return self.convert_to_unity_format(
                    m_bestNodeId, m_maxGoodMatches, m_confidence, m_bestPose
                )

        return {
            "status": "Failed",
            "confidence": "0.0%",
            "message": "Low confidence or no match",
        }

    def convert_to_unity_format(self, node_id, matches, confidence, pose_mtx):
        """
        将位姿矩阵转换为Unity坐标系 (左手系，Y向上)。
        """
        # 提取旋转和平移
        rot_mtx = pose_mtx[:3, :3]
        trans = pose_mtx[:3, 3]

        # RTABMap (OpenCV) -> Unity 坐标系转换逻辑:
        # OpenCV: X右, Y下, Z前 (右手系)
        # Unity: X右, Y上, Z前 (左手系)
        unity_pos = {
            "x": round(float(trans[0]), 3),
            "y": round(float(-trans[1]), 3), # Y 轴反转
            "z": round(float(trans[2]), 3),
        }

        # 旋转矩阵转换
        r = R.from_matrix(rot_mtx)
        q = r.as_quat() # [x, y, z, w]
        
        # 适配 Unity 四元数坐标系变换
        unity_quat = {
            "x": round(float(-q[0]), 4),
            "y": round(float(q[1]), 4),
            "z": round(float(-q[2]), 4),
            "w": round(float(q[3]), 4),
        }

        return {
            "status": "Success",
            "node_id": node_id,
            "confidence": f"{confidence}%",
            "match_count": matches,
            "unity_pos": unity_pos,
            "unity_quat": unity_quat,
        }

# --- 使用示例 ---
if __name__ == "__main__":
    # 请确保 map.db 和图片路径正确
    relocator = RTABMapRelocator("map.db")
    result = relocator.find_location(r"Scripts\Temp\image_1770089738128.jpg")
    print(result)