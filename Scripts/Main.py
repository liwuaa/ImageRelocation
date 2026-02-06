import os
import cv2
import time
import base64
import hashlib
import sqlite3
import numpy as np

from scipy.spatial.transform import Rotation as R
from flask import Flask, request, jsonify
from threading import Lock
from collections import OrderedDict
from AlgorithmClass import (
    AdvancedGeometryStrategy,
    BaseMatchingStrategy,
    OriginalRansacStrategy,
)


# --- 工具类 ---
class LRUCache:
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value


# --- 主类重构 ---


class OptimizedRTABMapRelocator:
    def __init__(self, db_path: str, strategy: BaseMatchingStrategy) -> None:
        self.m_dbPath = db_path
        self.m_strategy = strategy  # 运行时选择的策略
        self.m_orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2)  # type: ignore
        self.m_targetSize = None
        self.map_features_cache = {}
        self.query_cache = LRUCache(capacity=20)
        self.db_lock = Lock()
        self.load_map_features()

    def load_map_features(self):
        print(f"Loading map with {type(self.m_strategy).__name__}...")
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
                    "keypoints": kp_m,
                    "descriptors": des_m,
                    "pose": np.frombuffer(pose_blob, dtype=np.float32).reshape(3, 4),
                }
        conn.close()

    def _get_db_image_format(self, cursor):
        cursor.execute("SELECT image FROM Data LIMIT 1")
        row = cursor.fetchone()
        return cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_GRAYSCALE).shape[::-1] if row else None  # type: ignore

    def find_location_from_bytes(self, image_bytes) -> dict:
        image_hash = hashlib.md5(image_bytes).hexdigest()
        cached = self.query_cache.get(image_hash)
        if cached:
            return cached

        with self.db_lock:
            if self.m_targetSize is None:
                conn = sqlite3.connect(self.m_dbPath)
                self.m_targetSize = self._get_db_image_format(conn.cursor())
                conn.close()

            nparr = np.frombuffer(image_bytes, np.uint8)
            raw_query = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if raw_query is None:
                return {"status": "Failed", "message": "Decode error"}
            if self.m_targetSize:
                raw_query = cv2.resize(raw_query, self.m_targetSize)
            query_img = cv2.equalizeHist(raw_query)
            kp_q, des_q = self.m_orb.detectAndCompute(query_img, None)
            if des_q is None:
                return {"status": "Failed", "message": "No features"}

            m_bestNodeId, m_maxInliers, m_bestPose, m_bestGoodMatches = -1, 0, None, []

            # 核心算法调用：通过策略模式执行
            for node_id, data in self.map_features_cache.items():
                inliers, matches, _ = self.m_strategy.process(
                    kp_q, des_q, data, query_img.shape
                )
                if inliers > m_maxInliers:
                    m_maxInliers, m_bestNodeId, m_bestPose, m_bestGoodMatches = (
                        inliers,
                        node_id,
                        data["pose"],
                        matches,
                    )

        m_confidence = min(round((m_maxInliers / 40.0) * 100, 2), 100.0)
        print(
            f"DEBUG [{type(self.m_strategy).__name__}]: Best Node {m_bestNodeId} | Inliers: {m_maxInliers}"
        )

        result = self.convert_to_unity_format(
            m_bestNodeId, m_maxInliers, m_confidence, m_bestPose
        )

        # 统一业务逻辑状态判定
        if m_bestNodeId != -1 and m_maxInliers >= 12:
            result.update({"status": "Success", "message": "匹配成功"})
        elif m_bestNodeId != -1 and m_maxInliers > 0:
            result.update({"status": "LowConfidence", "message": "置信度较低"})
        else:
            result.update({"status": "Failed", "message": "匹配失败"})

        result.update({"query_keypoints": kp_q, "best_matches": m_bestGoodMatches})
        self.query_cache.put(image_hash, result)
        return result

    def convert_to_unity_format(self, node_id, matches, confidence, pose_mtx):
        if pose_mtx is None:
            return {
                "node_id": -1,
                "confidence": "0%",
                "match_count": 0,
                "unity_pos": {},
                "unity_quat": {},
            }
        rot_mtx, trans = pose_mtx[:3, :3], pose_mtx[:3, 3]
        unity_pos = {
            "x": round(float(trans[0]), 3),
            "y": round(float(-trans[1]), 3),
            "z": round(float(trans[2]), 3),
        }
        q = R.from_matrix(rot_mtx).as_quat()
        unity_quat = {
            "x": round(float(-q[0]), 4),
            "y": round(float(q[1]), 4),
            "z": round(float(-q[2]), 4),
            "w": round(float(q[3]), 4),
        }
        return {
            "node_id": node_id,
            "confidence": f"{confidence}%",
            "match_count": matches,
            "unity_pos": unity_pos,
            "unity_quat": unity_quat,
        }


# --- Server Context ---

app = Flask(__name__)
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "Temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# !!! 启动配置在这里选择策略 !!!
# 选项 A: OriginalRansacStrategy()  - 原始单应性矩阵版
# 选项 B: AdvancedGeometryStrategy() - 基础矩阵+密度校验版
# CURRENT_STRATEGY = OriginalRansacStrategy()
CURRENT_STRATEGY = AdvancedGeometryStrategy()
relocator = OptimizedRTABMapRelocator("2_5.db", CURRENT_STRATEGY)


def save_matched_images_with_features(
    query_image_data, map_node_id, query_keypoints, matches
):
    
    conn = sqlite3.connect(relocator.m_dbPath)
    cursor = conn.cursor()
    cursor.execute("SELECT image FROM Data WHERE id = ?", (map_node_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return
    # 1. 解码原始图像
    query_img = cv2.imdecode(np.frombuffer(query_image_data, np.uint8), cv2.IMREAD_COLOR)
    # 从数据库读取地图图像 (参考 relocator.m_dbPath)
    map_img = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)

    if query_img is None or map_img is None:
        return

    # 获取地图节点缓存的特征点
    map_kp = relocator.map_features_cache[map_node_id]["keypoints"]

    # --- 第一阶段：在各自的原始图片上标记特征点 ---
    
    # 复制图片，避免修改原始像素数据
    q_canvas = query_img.copy()
    m_canvas = map_img.copy()

    for match in matches:
        # 获取匹配对的原始坐标
        q_pt_raw = (int(query_keypoints[match.queryIdx].pt[0]), int(query_keypoints[match.queryIdx].pt[1]))
        m_pt_raw = (int(map_kp[match.trainIdx].pt[0]), int(map_kp[match.trainIdx].pt[1]))

        # 分别绘制红色(Query)和蓝色(Map)点
        cv2.circle(q_canvas, q_pt_raw, 5, (0, 0, 255), -1)
        cv2.circle(m_canvas, m_pt_raw, 5, (255, 0, 0), -1)

    # --- 第二阶段：水平拼接处理过的图片 ---
    
    h1, w1 = q_canvas.shape[:2]
    h2, w2 = m_canvas.shape[:2]
    
    # 保持高度统一（以两图中的最大高度为基准）
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

    # 放置图片
    vis[:h1, :w1] = q_canvas
    vis[:h2, w1:w1+w2] = m_canvas

    # --- 第三阶段：绘制跨图连接线 ---

    for match in matches:
        # 左图坐标保持不变
        q_pt = (int(query_keypoints[match.queryIdx].pt[0]), int(query_keypoints[match.queryIdx].pt[1]))
        # 右图坐标需要加上左图宽度偏移 w1
        m_pt = (int(map_kp[match.trainIdx].pt[0]) + w1, int(map_kp[match.trainIdx].pt[1]))

        # 绘制黄色匹配线 (AA抗锯齿)
        cv2.line(vis, q_pt, m_pt, (0, 255, 255), 1, cv2.LINE_AA)

    # --- 辅助信息绘制 ---
    # 使用半透明黑色矩形背景增强文字可读性
    cv2.rectangle(vis, (0, 0), (220, 80), (0, 0, 0), -1)
    cv2.putText(vis, f"Matches: {len(matches)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(vis, f"Node ID: {map_node_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 保存结果
    path = os.path.join(TEMP_DIR, f"matched_{map_node_id}_{int(time.time()*1000)}.jpg")
    cv2.imwrite(path, vis)

@app.route("/get_position", methods=["POST"])
def get_position():
    try:
        print("-------------***********----------------")
        data = request.get_json()
        image_data = base64.b64decode(data["image"])

        # 调用算法策略获取结果
        result = relocator.find_location_from_bytes(image_data)
        print(result["status"])

        # 处理图片匹配结果与可视化保存
        if result["status"] == "Success":
            # 仅在匹配成功时保存匹配图片
            if (
                "node_id" in result
                and "query_keypoints" in result
                and "best_matches" in result
            ):
                save_matched_images_with_features(
                    image_data,
                    result["node_id"],
                    result["query_keypoints"],
                    result["best_matches"],
                )
            print(
                f"SUCCESS: NodeID {result.get('node_id', 'N/A')} | Conf {result.get('confidence', 'N/A')}"
            )
        elif result["status"] == "LowConfidence":
            print(
                f"LOW CONFIDENCE: Conf {result.get('confidence', 'N/A')} | Inliers: {result.get('match_count', 'N/A')}"
            )
        else:
            print(
                f"FAILED: {result.get('message')} | Inliers: {result.get('match_count', 'N/A')}"
            )

        # --- 还原回你原来的响应格式 ---
        newResult = {
            "status": result["status"],  # 业务状态（Success、LowConfidence 或 Failed）
            "confidence": result[
                "confidence"
            ],  # 这里 result['confidence'] 已经是带 % 的字符串了
            "match_count": result.get("match_count", 0),
            "unity_pos": result.get("unity_pos", {}),
            "unity_quat": result.get("unity_quat", {}),
        }

        print("*********************************\n")
        # 接口状态总是200 OK，业务状态由 result 中的 status 字段表示
        # message 字段包含序列化后的业务数据字符串
        return jsonify({"status": result["status"], "message": str(newResult)})

    except Exception as e:
        # 发生异常时返回错误状态
        print(f"Exception: {str(e)}")
        error_info = {
            "status": "Failed",
            "confidence": "0%",
            "match_count": 0,
            "unity_pos": {},
            "unity_quat": {},
            "message": f"Exception occurred: {str(e)}",
        }
        print("*********************************\n")
        return jsonify({"status": "Failed", "message": str(error_info)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
