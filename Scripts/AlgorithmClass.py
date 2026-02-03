import cv2
import numpy as np



class BaseMatchingStrategy:
    """基础匹配算法接口"""
    def process(self, kp_q, des_q, map_data, query_img_shape):
        raise NotImplementedError

class OriginalRansacStrategy(BaseMatchingStrategy):
    """你原本的单应性矩阵 RANSAC 算法"""
    def process(self, kp_q, des_q, map_data, query_img_shape):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_q, map_data["descriptors"], k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        
        if len(good) < 15: return 0, [], None
        
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) # type: ignore
        dst_pts = np.float32([map_data["keypoints"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)# type: ignore
        
        # 使用单应性矩阵 (原算法)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is not None:
            inliers = int(np.sum(mask))
            best_matches = [good[i] for i in range(len(mask)) if mask[i]]
            return inliers, best_matches, mask
        return 0, [], None

class AdvancedGeometryStrategy(BaseMatchingStrategy):
    """新增：基础矩阵 + 分布密度校验 算法"""
    def process(self, kp_q, des_q, map_data, query_img_shape):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_q, map_data["descriptors"], k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        
        if len(good) < 15: return 0, [], None
        
        src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)# type: ignore
        dst_pts = np.float32([map_data["keypoints"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)# type: ignore
        
        # 1. 使用基础矩阵校验 (更适合 3D 办公场景)
        _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0)
        
        if mask is not None:
            current_inliers = int(np.sum(mask))
            inlier_mask = mask.ravel() == 1
            
            # 2. 分布密度校验
            if current_inliers > 10:
                pts_in_query = src_pts[inlier_mask]
                x_coords, y_coords = pts_in_query[:, 0, 0], pts_in_query[:, 0, 1]
                width = np.max(x_coords) - np.min(x_coords)
                height = np.max(y_coords) - np.min(y_coords)
                area_ratio = (width * height) / (query_img_shape[0] * query_img_shape[1])
                
                # 分布太集中则惩罚权重
                if area_ratio < 0.06:
                    current_inliers = int(current_inliers * 0.5)
            
            best_matches = [good[i] for i in range(len(mask)) if inlier_mask[i]]
            return current_inliers, best_matches, mask
        return 0, [], None
