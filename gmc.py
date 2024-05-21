import numpy as np
import cv2
import functools

def motion_compensate_for_nbb(blocks, ref0_image, ref1_image):
    compensated_blocks = []
    for (y, x), block in blocks:
        # Find corresponding blocks in reference images
        ref0_block = ref0_image[y:y+block.shape[0], x:x+block.shape[1]]
        ref1_block = ref1_image[y:y+block.shape[0], x:x+block.shape[1]]
        
        # Apply feature matching for motion compensation
        compensated_block = feature_matching_for_nbb(block, ref0_block, ref1_block)
        compensated_blocks.append((compensated_block, (y, x)))
    return compensated_blocks

def feature_matching_for_nbb(blk_target, blk_ref0, blk_ref1):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp_target, des_target = orb.detectAndCompute(blk_target, None)
    kp_ref0, des_ref0 = orb.detectAndCompute(blk_ref0, None)
    kp_ref1, des_ref1 = orb.detectAndCompute(blk_ref1, None)

    if des_target is None or (des_ref0 is None and des_ref1 is None):
        # print("No descriptors found, returning original block.")
        if des_ref0 is None:
            return blk_ref1
        else:
            return blk_ref0

    # Initiate bf matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches0, matches1 = [],[]
    if des_ref0 is not None:
        matches0 = bf.match(des_target, des_ref0) 
        matches0 = sorted(matches0, key=lambda x: x.distance)

    if des_ref1 is not None:
        matches1 = bf.match(des_target, des_ref1) 
        matches1 = sorted(matches1, key=lambda x: x.distance)

    # Calculate total distance for matches0 and matches1
    if matches0 is not None:
        total_distance0 = sum([m.distance for m in matches0])
    if matches1 is not None:
        total_distance1 = sum([m.distance for m in matches1])

    # Calculate total distance for matches0 and matches1
    total_distance0 = sum([m.distance for m in matches0])
    total_distance1 = sum([m.distance for m in matches1])

    # Choose the matches with the lower total distance (higher similarity)
    if total_distance0 < total_distance1:
        # print('ref0 choosen.')
        chosen_matches = matches0
        kp_ref = kp_ref0
        blk_ref = blk_ref0
    else:
        # print('ref1 choosen.')
        chosen_matches = matches1
        kp_ref = kp_ref1
        blk_ref = blk_ref1

    dst_pts = np.float32([kp_target[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_ref[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    
    if(len(dst_pts) > 8):
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        compensated_block = cv2.warpAffine(blk_ref, M, (blk_ref0.shape[1], blk_ref0.shape[0]))
    else:
        # print("Not enough matche points, returning original block.")
        compensated_block = blk_ref

    return compensated_block    

def feature_matching(blk_target, blk_ref0, blk_ref1, idx_ref0, idx_ref1):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp_target, des_target = orb.detectAndCompute(blk_target, None)
    kp_ref0, des_ref0 = orb.detectAndCompute(blk_ref0, None)
    kp_ref1, des_ref1 = orb.detectAndCompute(blk_ref1, None)

    if des_target is None or (des_ref0 is None and des_ref1 is None):
        # print("No descriptors found, returning original block.")
        if des_ref0 is None:
            return blk_ref1
        else:
            return blk_ref0

    # Initiate bf matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches0, matches1 = [],[]
    if idx_ref0 != -1 and des_ref0 is not None:
        matches0 = bf.match(des_target, des_ref0) 
        matches0 = sorted(matches0, key=lambda x: x.distance)

    if idx_ref1 != -1 and des_ref1 is not None:
        matches1 = bf.match(des_target, des_ref1) 
        matches1 = sorted(matches1, key=lambda x: x.distance)

    # Calculate total distance for matches0 and matches1
    if matches0 is not None:
        total_distance0 = sum([m.distance for m in matches0])
    if matches1 is not None:
        total_distance1 = sum([m.distance for m in matches1])

    # Choose the matches with the lower total distance (higher similarity)
    if total_distance0 < total_distance1:
        # print('ref0 choosen.')
        chosen_matches = matches0
        kp_ref = kp_ref0
        blk_ref = blk_ref0
    else:
        # print('ref1 choosen.')
        chosen_matches = matches1
        kp_ref = kp_ref1
        blk_ref = blk_ref1

    dst_pts = np.float32([kp_target[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_ref[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    
    if(len(dst_pts) > 8):
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        compensated_block = cv2.warpAffine(blk_ref, M, (blk_ref0.shape[1], blk_ref0.shape[0]))

    else:
        # print("Not enough matche points, returning original block.")
        compensated_block = blk_ref

    return compensated_block    
