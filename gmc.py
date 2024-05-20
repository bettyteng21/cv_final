import numpy as np
import cv2
import functools


def feature_matching(blk_target, blk_ref0, blk_ref1, flow):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp_target, des_target = orb.detectAndCompute(blk_target, None)
    kp_ref0, des_ref0 = orb.detectAndCompute(blk_ref0, None)
    kp_ref1, des_ref1 = orb.detectAndCompute(blk_ref1, None)

    if des_target is None or des_ref0 is None or des_ref1 is None:
        print("No descriptors found, returning original block.")
        if des_ref0 is None:
            return blk_ref1
        else:
            return blk_ref0

    # Initiate bf matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches0 = bf.match(des_target, des_ref0) 
    matches0 = sorted(matches0, key=lambda x: x.distance)

    matches1 = bf.match(des_target, des_ref1) 
    matches1 = sorted(matches1, key=lambda x: x.distance)

    # Calculate total distance for matches0 and matches1
    total_distance0 = sum([m.distance for m in matches0])
    total_distance1 = sum([m.distance for m in matches1])

    # Choose the matches with the lower total distance (higher similarity)
    if total_distance0 < total_distance1:
        print('ref0 choosen.')
        chosen_matches = matches0
        kp_ref = kp_ref0
        blk_ref = blk_ref0
    else:
        print('ref1 choosen.')
        chosen_matches = matches1
        kp_ref = kp_ref1
        blk_ref = blk_ref1

    dst_pts = np.float32([kp_target[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_ref[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    
    if(len(dst_pts) > 8):
        # 計算仿射變換矩陣
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        # 對block進行仿射變換補償
        compensated_block = cv2.warpAffine(blk_ref, M, (blk_ref0.shape[1], blk_ref0.shape[0]))

        # # 使用光流進行進一步補償
        # h, w = blk_target.shape
        # flow_x = flow[:h, :w, 0]
        # flow_y = flow[:h, :w, 1]
        # grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # # 確保 grid_x 和 grid_y 是 float32 類型
        # map_x = (grid_x + flow_x).astype(np.float32)
        # map_y = (grid_y + flow_y).astype(np.float32)

        # compensated_block = cv2.remap(compensated_block, map_x, map_y, cv2.INTER_LINEAR)
    else:
        print("Not enough matche points, returning original block.")
        compensated_block = blk_ref

    return compensated_block    
