import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd
from PIL import Image

def hexagon_based_search(ref_frame, target_block, y, x, block_size, search_range):
    directions = [(0, -1), (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1)]  # 6 directions for hexagon search
    min_mse = float('inf')
    best_match = (0, 0)
    for dy, dx in directions:
        y0, x0 = y + dy, x + dx
        if 0 <= y0 < ref_frame.shape[0] - block_size and 0 <= x0 < ref_frame.shape[1] - block_size:
            ref_block = ref_frame[y0:y0 + block_size, x0:x0 + block_size]
            mse = np.mean((target_block - ref_block) ** 2)
            if mse < min_mse:
                min_mse = mse
                best_match = (dy, dx)
    return best_match

def block_matching(ref_frame, target_block, y, x, block_size, search_range):
    min_mse = float('inf')
    best_match = (0, 0)
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            y0, x0 = y + dy, x + dx
            if 0 <= y0 < ref_frame.shape[0] - block_size and 0 <= x0 < ref_frame.shape[1] - block_size:
                ref_block = ref_frame[y0:y0 + block_size, x0:x0 + block_size]
                mse = np.mean((target_block - ref_block) ** 2)
                if mse < min_mse:
                    min_mse = mse
                    best_match = (dy, dx)
    return best_match


def motion_estimation(ref_frame0, ref_frame1, target_frame, _type, block_size=16, search_range=7):
    motion_vectors_ref0_to_target = []
    motion_vectors_ref1_to_target = []

    for y in range(0, target_frame.shape[0], block_size):
        for x in range(0, target_frame.shape[1], block_size):
            target_block = target_frame[y:y + block_size, x:x + block_size]
            if _type == 'full_search':    
                mv0 = block_matching(ref_frame0, target_block, y, x, block_size, search_range)
                mv1 = block_matching(ref_frame1, target_block, y, x, block_size, search_range)
            elif _type == 'hexagon_search':
                mv0 = hexagon_based_search(ref_frame0, target_block, y, x, block_size, search_range)
                mv1 = hexagon_based_search(ref_frame1, target_block, y, x, block_size, search_range)
            motion_vectors_ref0_to_target.append(mv0)
            motion_vectors_ref1_to_target.append(mv1)

    return np.array(motion_vectors_ref0_to_target), np.array(motion_vectors_ref1_to_target)


def global_motion_compensation(ref_frame0, ref_frame1, motion_vectors, block_size=16):
    predicted_frame = np.zeros_like(ref_frame0, dtype=np.float32)
    weights = np.zeros_like(ref_frame0, dtype=np.float32)

    for i, ((dy0, dx0), (dy1, dx1)) in enumerate(zip(motion_vectors[0], motion_vectors[1])):
        y = (i // (ref_frame0.shape[1] // block_size)) * block_size
        x = (i % (ref_frame0.shape[1] // block_size)) * block_size

        y0, x0 = y + dy0, x + dx0
        y1, x1 = y + dy1, x + dx1

        if 0 <= y0 < ref_frame0.shape[0] - block_size and 0 <= x0 < ref_frame0.shape[1] - block_size:
            ref_block0 = ref_frame0[y0:y0 + block_size, x0:x0 + block_size]
            predicted_frame[y:y + block_size, x:x + block_size] += ref_block0
            weights[y:y + block_size, x:x + block_size] += 1

        if 0 <= y1 < ref_frame1.shape[0] - block_size and 0 <= x1 < ref_frame1.shape[1] - block_size:
            ref_block1 = ref_frame1[y1:y1 + block_size, x1:x1 + block_size]
            predicted_frame[y:y + block_size, x:x + block_size] += ref_block1
            weights[y:y + block_size, x:x + block_size] += 1

    non_zero_weights = weights > 0
    predicted_frame[non_zero_weights] /= weights[non_zero_weights]
    predicted_frame = np.clip(predicted_frame, 0, 255).astype(np.uint8)

    return predicted_frame


def sift_feature_matching(target_img, ref_img1, ref_img2):

    sift = cv2.SIFT_create()

    keypoints_target, descriptors_target = sift.detectAndCompute(target_img, None)
    keypoints_ref1, descriptors_ref1 = sift.detectAndCompute(ref_img1, None)
    keypoints_ref2, descriptors_ref2 = sift.detectAndCompute(ref_img2, None)

    bf = cv2.BFMatcher()

    matches_ref1 = bf.knnMatch(descriptors_target, descriptors_ref1, k=2)
    matches_ref2 = bf.knnMatch(descriptors_target, descriptors_ref2, k=2)

    # 提取良好的匹配點
    good_matches_ref1 = []
    for m, n in matches_ref1:
        if m.distance < 0.7 * n.distance:
            good_matches_ref1.append(m)

    good_matches_ref2 = []
    for m, n in matches_ref2:
        if m.distance < 0.7 * n.distance:
            good_matches_ref2.append(m)

    # 收集匹配點的坐標
    src_pts_ref1 = np.float32([keypoints_target[m.queryIdx].pt for m in good_matches_ref1]).reshape(-1, 1, 2)
    dst_pts_ref1 = np.float32([keypoints_ref1[m.trainIdx].pt for m in good_matches_ref1]).reshape(-1, 1, 2)

    src_pts_ref2 = np.float32([keypoints_target[m.queryIdx].pt for m in good_matches_ref2]).reshape(-1, 1, 2)
    dst_pts_ref2 = np.float32([keypoints_ref2[m.trainIdx].pt for m in good_matches_ref2]).reshape(-1, 1, 2)

    # 合併特徵匹配結果
    src_pts = np.concatenate((src_pts_ref1, src_pts_ref2), axis=0)
    dst_pts = np.concatenate((dst_pts_ref1, dst_pts_ref2), axis=0)
    
    return src_pts, dst_pts


def brisk_feature_matching(target_img, ref_img1, ref_img2):

    brisk = cv2.BRISK_create()
    keypoints_ref1, descriptors_ref1 = brisk.detectAndCompute(ref_img1, None)
    keypoints_ref2, descriptors_ref2 = brisk.detectAndCompute(ref_img2, None)
    keypoints_target, descriptors_target = brisk.detectAndCompute(target_img, None)
    
    bf = cv2.BFMatcher()

    matches_ref1 = bf.knnMatch(descriptors_ref1, descriptors_target, k=2)
    matches_ref2 = bf.knnMatch(descriptors_ref2, descriptors_target, k=2)

    # 應用比值測試以確定好的匹配
    good_matches_ref1 = []
    for m, n in matches_ref1:
        if m.distance < 0.75 * n.distance:
            good_matches_ref1.append(m)

    good_matches_ref2 = []
    for m, n in matches_ref2:
        if m.distance < 0.75 * n.distance:
            good_matches_ref2.append(m)

    # 提取匹配特徵點的坐標
    src_pts_ref1 = np.float32([keypoints_ref1[m.queryIdx].pt for m in good_matches_ref1]).reshape(-1, 1, 2)
    dst_pts_ref1 = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches_ref1]).reshape(-1, 1, 2)

    src_pts_ref2 = np.float32([keypoints_ref2[m.queryIdx].pt for m in good_matches_ref2]).reshape(-1, 1, 2)
    dst_pts_ref2 = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches_ref2]).reshape(-1, 1, 2)

    # 合併特徵匹配結果
    src_pts = np.concatenate((src_pts_ref1, src_pts_ref2), axis=0)
    dst_pts = np.concatenate((dst_pts_ref1, dst_pts_ref2), axis=0)

    return src_pts, dst_pts

def apply_global_motion_compensation_Affine(target_img, ref_img, src_pts, dst_pts):
    
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
    compensated_img = cv2.warpAffine(ref_img, M, (target_img.shape[1], target_img.shape[0]))

    return compensated_img

def apply_global_motion_compensation_Perspective(target_img, ref_img, src_pts, dst_pts):
    
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    compensated_img = cv2.warpPerspective(ref_img, M, (target_img.shape[1], target_img.shape[0]))

    return compensated_img

def apply_dense_optical_flow_compensation(target_img, prev_img, next_img):
    
    flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    # 創建一個空的補償圖像
    compensated_img = np.zeros_like(target_img)
    # 對每個像素應用光流場中的位移
    for y in range(h):
        for x in range(w):
            dx, dy = flow[y, x]
            new_x = int(x + dx)
            new_y = int(y + dy)

            # 確保新座標在圖像範圍內
            if 0 <= new_x < w and 0 <= new_y < h:
                compensated_img[new_y, new_x] = target_img[y, x]

    return compensated_img
