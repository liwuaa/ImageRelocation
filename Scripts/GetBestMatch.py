import sqlite3
import cv2
import numpy as np
import zlib
import io

def get_best_match(db_path, query_image_path):
    # 1. 连接 RTAB-Map 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. 读取传入的照片并提取特征 (建议使用与 RTAB-Map 相同的算法，如 ORB)
    query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=1000) # type: ignore
    kp_query, des_query = orb.detectAndCompute(query_img, None)

    if des_query is None:
        return "无法提取查询图像的特征"

    # 3. 从数据库中检索所有关键帧图像
    # 注意：RTAB-Map 的 Data 表中图像通常是经过压缩的
    cursor.execute("SELECT id, image FROM Data")
    all_data = cursor.fetchall()

    best_node_id = -1
    max_good_matches = 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print(f"正在地图中检索匹配帧...")

    for node_id, img_blob in all_data:
        if img_blob is None: continue
        
        # 解压图像数据 (RTAB-Map 默认可能使用 zlib 压缩或直接存储字节流)
        try:
            nparr = np.frombuffer(img_blob, np.uint8)
            map_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # 提取地图帧特征
            kp_map, des_map = orb.detectAndCompute(map_img, None)
            if des_map is None: continue

            # 特征匹配
            matches = bf.match(des_query, des_map)
            good_matches = len(matches)

            if good_matches > max_good_matches:
                max_good_matches = good_matches
                best_node_id = node_id
        except Exception as e:
            continue

    if best_node_id != -1:
        # 4. 获取最匹配节点的位姿信息 (Pose)
        # Pose 存储在 Node 表中，为 3x4 的变换矩阵 (float 字节流)
        cursor.execute("SELECT pose FROM Node WHERE id=?", (best_node_id,))
        pose_blob = cursor.fetchone()[0]
        pose = np.frombuffer(pose_blob, dtype=np.float32).reshape(3, 4)
        
        conn.close()
        return {
            "node_id": best_node_id,
            "match_count": max_good_matches,
            "pose_matrix": pose.tolist()
        }
    else:
        conn.close()
        return "未找到匹配位置"

if __name__ == "__main__":
# 使用示例
    db_file = "map.db"
    query_photo = "Scripts\Temp\image_1770089738128.jpg"
    result = get_best_match(db_file, query_photo)
    print(result)