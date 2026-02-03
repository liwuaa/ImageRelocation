"""
Unity图像位置服务 - 优化版
用于接收Unity发送的图像数据，计算其在地图中的位置，并返回位置信息
优化版本包含缓存机制和性能优化
"""
import cv2
import numpy as np
import sqlite3
from scipy.spatial.transform import Rotation as R
from flask import Flask, request, jsonify
import base64
import os
from threading import Lock
from collections import OrderedDict
import hashlib


class LRUCache:
    """简单LRU缓存实现，用于存储最近的查询结果"""
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            # 移动到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            # 更新现有项
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # 删除最久未使用的项
            self.cache.popitem(last=False)
        
        self.cache[key] = value


class OptimizedRTABMapRelocator:
    """
    优化版RTABMap重定位器类，加入了缓存机制和性能优化
    """
    def __init__(self, db_path: str) -> None:
        self.m_dbPath = db_path
        # 增加特征点以提高鲁棒性
        self.m_orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2)   # type: ignore
        # 注意：使用 KNN 时 crossCheck 必须为 False
        self.m_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.m_targetSize: tuple[int, int] | None = None
        self.m_totalQueryFeatures = 0
        
        # 加载数据库中的所有映射图像特征
        self.map_features_cache = {}
        self.load_map_features()
        
        self.query_cache = LRUCache(capacity=20)
        self.db_lock = Lock()

    def load_map_features(self):
        """预加载数据库中的映射图像特征"""
        print("Loading map features...")
        conn = sqlite3.connect(self.m_dbPath)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT Node.id, Data.image, Node.pose FROM Node JOIN Data ON Node.id = Data.id"
        )
        rows = cursor.fetchall()
        
        for node_id, img_blob, pose_blob in rows:
            if not img_blob or not pose_blob:
                continue
                
            nparr = np.frombuffer(img_blob, np.uint8)
            map_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if map_img is None:
                continue
                
            map_img = cv2.equalizeHist(map_img)
            
            kp_m, des_m = self.m_orb.detectAndCompute(map_img, None)
            if des_m is not None:
                self.map_features_cache[node_id] = {
                    'descriptors': des_m,
                    'pose': np.frombuffer(pose_blob, dtype=np.float32).reshape(3, 4)
                }
        
        conn.close()
        print(f"Loaded features for {len(self.map_features_cache)} map nodes")

    def find_location_from_bytes(self, image_bytes) -> dict:
        """
        核心算法优化版：从字节流中查找位置
        """
        image_hash = hashlib.md5(image_bytes).hexdigest()
        cached_result = self.query_cache.get(image_hash)
        if cached_result is not None:
            return cached_result

        with self.db_lock:
            # 获取数据库图像尺寸格式（仅用于 Resize）
            if self.m_targetSize is None:
                conn = sqlite3.connect(self.m_dbPath)
                self.m_targetSize = self._get_db_image_format(conn.cursor())
                conn.close()

            nparr = np.frombuffer(image_bytes, np.uint8)
            raw_query = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if raw_query is None:
                return {"status": "Failed", "message": "Error: Image not decoded"}

            if self.m_targetSize:
                raw_query = cv2.resize(raw_query, self.m_targetSize, interpolation=cv2.INTER_AREA)
            query_img = cv2.equalizeHist(raw_query)

            # 提取查询图特征
            kp_q, des_q = self.m_orb.detectAndCompute(query_img, None)
            if des_q is None:
                return {"status": "Failed", "message": "No features found"}
            
            self.m_totalQueryFeatures = len(kp_q)

            m_bestNodeId = -1
            m_maxGoodMatches = 0
            m_bestPose = None

            # --- 优化：高质量匹配循环 ---
            for node_id, features_data in self.map_features_cache.items():
                des_m = features_data['descriptors']
                
                # 使用 knnMatch (k=2) 以便执行 Ratio Test
                matches = self.m_bf.knnMatch(des_q, des_m, k=2)
                
                # Lowe's ratio test 筛选
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                current_good_count = len(good_matches)
                if current_good_count > m_maxGoodMatches:
                    m_maxGoodMatches = current_good_count
                    m_bestNodeId = node_id
                    m_bestPose = features_data['pose']

        # --- 优化：科学置信度计算 ---
        # 1. 基础匹配率
        match_ratio = m_maxGoodMatches / self.m_totalQueryFeatures
        # 2. 数量惩罚权重：好匹配点少于40个时置信度按比例下降
        count_weight = min(m_maxGoodMatches / 40.0, 1.0)
        
        # 放大系数设为 300，即好匹配率达 33.3% 且数量足时为 100%
        m_confidence = (match_ratio * 300) * count_weight * 100
        m_confidence = min(round(m_confidence, 2), 100.0)

        if m_bestNodeId != -1 and m_confidence > 12.0: # 门槛略微提高
            result = self.convert_to_unity_format(
                m_bestNodeId, m_maxGoodMatches, m_confidence, m_bestPose
            )
        else:
            result = {
                "status": "Failed",
                "confidence": f"{m_confidence}%",
                "message": "Low confidence match",
            }

        self.query_cache.put(image_hash, result)
        return result
    def _get_db_image_format(self, cursor) -> tuple[int, int] | None:
        """
        获取数据库中图像的尺寸格式。

        参数:
            cursor: 数据库游标，用于执行SQL查询

        返回:
            tuple: 图像尺寸 (宽度, 高度)，如果无法获取则返回None
        """
        cursor.execute("SELECT image FROM Data LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            nparr = np.frombuffer(row[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            return img.shape[::-1] # type: ignore
        return None
    def convert_to_unity_format(self, node_id, matches, confidence, pose_mtx):
        """
        将给定的姿态矩阵转换为Unity引擎所需的格式，包括位置和四元数旋转信息。

        参数:
            node_id (str): 节点标识符，用于标识当前处理的节点。
            matches (int): 匹配数量，表示与目标对象匹配的特征点数量。
            confidence (float): 置信度，表示匹配结果的可信程度（通常以百分比形式表示）。
            pose_mtx (numpy.ndarray): 4x4的姿态矩阵，包含旋转和平移信息。

        返回:
            dict: 包含转换后的位置、旋转、置信度和匹配数量等信息的字典
        """

        # 提取旋转矩阵和平移向量
        rot_mtx = pose_mtx[:3, :3]
        trans = pose_mtx[:3, 3]

        # 将平移向量转换为Unity坐标系 (X, -Y, Z)
        unity_pos = {
            "x": round(float(trans[0]), 3),
            "y": round(float(-trans[1]), 3),
            "z": round(float(trans[2]), 3),
        }

        # 将旋转矩阵转换为四元数，并适配Unity坐标系
        r = R.from_matrix(rot_mtx)
        q = r.as_quat()
        unity_quat = {
            "x": round(float(-q[0]), 4),
            "y": round(float(q[1]), 4),
            "z": round(float(-q[2]), 4),
            "w": round(float(q[3]), 4),
        }

        # 构造并返回结果字典
        return {
            "status": "Success",
            "node_id": node_id,
            "confidence": f"{confidence}%",
            "match_count": matches,
            "unity_pos": unity_pos,
            "unity_quat": unity_quat,
        }


# 创建Flask应用
app = Flask(__name__)

# 初始化重定位器
relocator = OptimizedRTABMapRelocator("66.db")

@app.route('/get_position', methods=['POST'])
def get_position():
    """
    接收Unity发送的图像数据，返回位置信息
    """
    try:
        # 从请求中获取图像数据
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"status": "Failed", "message": "No image data received"})
        
        # 解码Base64图像数据
        image_data = base64.b64decode(data['image'])
            
        # 查找位置
        result = relocator.find_location_from_bytes(image_data)
        # 缓存图片
        try :
            save_image_to_temp(image_data,result['node_id'])
        except Exception as e:
            print(f"Error saving image: {str(e)}") 
        print(result)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "Failed", "message": f"Error processing request: {str(e)}"})


@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    return jsonify({"status": "Healthy"})


@app.route('/stats', methods=['GET'])
def stats():
    """
    获取服务统计信息
    """
    cache_size = len(relocator.query_cache.cache)
    total_nodes = len(relocator.map_features_cache)
    
    return jsonify({
        "cache_size": cache_size,
        "cached_items": list(relocator.query_cache.cache.keys())[-5:],  # 最近5个缓存项
        "total_map_nodes": total_nodes,
        "status": "Healthy"
    })

TEMP_DIR = os.path.join(os.path.dirname(__file__), 'Temp')
def save_image_to_temp(image_data, filename=None):
    """
    将图片数据保存到Temp文件夹
    
    参数:
        image_data (bytes): 图片的二进制数据
        filename (str, optional): 文件名，如果不提供则自动生成
    
    返回:
        str: 保存的文件路径
    """
    if filename is None:
        # 生成基于时间戳的唯一文件名
        import time
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
        filename = f"image_{timestamp}.jpg"
    
    file_path = os.path.join(TEMP_DIR, filename)
    
    # 保存图片数据
    with open(file_path, 'wb') as f:
        f.write(image_data)
    
    print(f"Image saved to: {file_path}")
    return file_path

if __name__ == '__main__':
    print("Starting Optimized Unity Image Position Service...")
    print(f"Loaded {len(relocator.map_features_cache)} map nodes")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)