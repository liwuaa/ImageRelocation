"""
Unity图像位置服务
用于接收Unity发送的图像数据，计算其在地图中的位置，并返回位置信息
"""

import cv2
import numpy as np
import sqlite3
import json
from scipy.spatial.transform import Rotation as R
from flask import Flask, request, jsonify
from io import BytesIO
import base64
import os


class RTABMapRelocator:
    """
    RTABMap重定位器类，用于从RTABMap数据库中查找图像位置并将位姿转换为Unity兼容格式。
    """
    
    def __init__(self, db_path: str) -> None:
        """
        初始化RTABMap重定位器。

        参数:
            db_path (str): RTABMap数据库文件的路径
        """
        self.m_dbPath = db_path
        self.m_orb = cv2.ORB_create(nfeatures=2000) # type: ignore
        self.m_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.m_targetSize: tuple[int, int] | None = None
        self.m_totalQueryFeatures = 0

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

    def find_location_from_bytes(self, image_bytes) -> dict:
        """
        从图像字节数据中查找与查询图像最匹配的位置。

        参数:
            image_bytes (bytes): 查询图像的字节数据

        返回:
            dict: 包含匹配结果的字典，如果成功则包含Unity格式的位姿信息，否则包含错误信息
        """
        conn = sqlite3.connect(self.m_dbPath)
        cursor = conn.cursor()

        self.m_targetSize = self._get_db_image_format(cursor)
        
        # 解码图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        raw_query = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if raw_query is None:
            return {"status": "Failed", "message": "Error: Image not decoded"}

        # 尺寸对齐与增强
        if self.m_targetSize:
            raw_query = cv2.resize(
                raw_query, self.m_targetSize, interpolation=cv2.INTER_AREA
            )
        query_img = cv2.equalizeHist(raw_query)

        # 提取特征
        kp_q, des_q = self.m_orb.detectAndCompute(query_img, None)
        self.m_totalQueryFeatures = len(kp_q) if kp_q else 1

        cursor.execute(
            "SELECT Node.id, Data.image, Node.pose FROM Node JOIN Data ON Node.id = Data.id"
        )
        rows = cursor.fetchall()

        m_bestNodeId = -1
        m_maxMatches = 0
        m_bestPose = None

        for node_id, img_blob, pose_blob in rows:
            if not img_blob or not pose_blob:
                continue

            nparr = np.frombuffer(img_blob, np.uint8)
            map_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            map_img = cv2.equalizeHist(map_img) # type: ignore

            kp_m, des_m = self.m_orb.detectAndCompute(map_img, None)
            if des_m is None:
                continue

            matches = self.m_bf.match(des_q, des_m)
            if len(matches) > m_maxMatches:
                m_maxMatches = len(matches)
                m_bestNodeId = node_id
                m_bestPose = np.frombuffer(pose_blob, dtype=np.float32).reshape(3, 4)

        conn.close()

        # --- 计算置信度百分比 ---
        # 逻辑：匹配数占查询图特征总数的比例，并乘以系数放大感官差异
        raw_score = (m_maxMatches / self.m_totalQueryFeatures) * 500
        m_confidence = min(round(raw_score, 2), 100.0)

        if m_bestNodeId != -1 and m_confidence > 10.0:  # 设定10%为最低门槛
            return self.convert_to_unity_format(
                m_bestNodeId, m_maxMatches, m_confidence, m_bestPose
            )
        else:
            return {
                "status": "Failed",
                "confidence": f"{m_confidence}%",
                "message": "Low confidence",
            }

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
            "confidence": f"{confidence}%",
            "match_count": matches,
            "unity_pos": unity_pos,
            "unity_quat": unity_quat,
        }


# 创建Flask应用
app = Flask(__name__)

# 初始化重定位器
relocator = RTABMapRelocator("map.db")


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


if __name__ == '__main__':
    print("Starting Unity Image Position Service...")
    app.run(host='0.0.0.0', port=5000, debug=False)