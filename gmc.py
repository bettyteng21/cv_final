import numpy as np
import cv2

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

def motion_estimation(ref_frame0, ref_frame1, target_frame, block_size=16, search_range=7):
    motion_vectors_ref0_to_target = []
    motion_vectors_ref1_to_target = []

    for y in range(0, target_frame.shape[0], block_size):
        for x in range(0, target_frame.shape[1], block_size):
            target_block = target_frame[y:y + block_size, x:x + block_size]
            mv0 = block_matching(ref_frame0, target_block, y, x, block_size, search_range)
            mv1 = block_matching(ref_frame1, target_block, y, x, block_size, search_range)
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

# for non-bounding-box blocks
def motion_compensate_for_nbb(blocks, ref0_image, ref1_image, lun_image):
    compensated_blocks = []
    for block,(y,x) in blocks:
        # Find corresponding blocks in reference images
        ref0_block = ref0_image[y:y+block.shape[0], x:x+block.shape[1]]
        ref1_block = ref1_image[y:y+block.shape[0], x:x+block.shape[1]]
        lun_block = lun_image[y:y+block.shape[0], x:x+block.shape[1]]
        
        # Apply feature matching for motion compensation
        compensated_block = feature_matching_for_nbb(block, ref0_block, ref1_block, lun_block)
        compensated_blocks.append((compensated_block, (y, x)))
    return compensated_blocks

# for non-bounding-box blocks
def feature_matching_for_nbb(blk_target, blk_ref0, blk_ref1, blk_lun):
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(300)
    kp_target, des_target = surf.detectAndCompute(blk_target,None)
    kp_ref0, des_ref0 = surf.detectAndCompute(blk_ref0,None)
    kp_ref1, des_ref1 = surf.detectAndCompute(blk_ref1,None)

    if des_target is None or (des_ref0 is None and des_ref1 is None):
        if des_ref0 is None:
            return blk_ref1
        else:
            return blk_ref0

    # Initiate bf matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
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

    total_distance0 = sum([m.distance for m in matches0])
    total_distance1 = sum([m.distance for m in matches1])

    # Choose the matches with the lower total distance (higher similarity)
    if total_distance0 < total_distance1:
        chosen_matches = matches0
        kp_ref = kp_ref0
        blk_ref = blk_ref0
    else:
        chosen_matches = matches1
        kp_ref = kp_ref1
        blk_ref = blk_ref1

    dst_pts = np.float32([kp_target[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_ref[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    
    result = np.zeros_like(blk_target)
    if(len(dst_pts) > 8):
        # calculate homography matrix if there's enough points
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        compensated_block = cv2.warpAffine(blk_ref, M, (blk_ref0.shape[1], blk_ref0.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))

        ret,thresh=cv2.threshold(compensated_block,1,255,0)
        mm,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        mask_bool = mm.astype(bool)
        result[mask_bool] = compensated_block[mask_bool]
        result[~mask_bool] = blk_lun[~mask_bool]
            
    else:
        result = blk_ref

    return result    

# for bounding-box blocks (detected objects)
def feature_matching(blk_target, blk_ref0, blk_ref1, idx_ref0, idx_ref1, blk_lun):
    '''
    When idx_ref0==-1 or idx_ref1==-1:
        for this target block, there's no corresponding obj in ref0/ref1 blocks.
        So we shouldn't do any match.
    '''

    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(300)
    kp_target, des_target = surf.detectAndCompute(blk_target,None)
    kp_ref0, des_ref0 = surf.detectAndCompute(blk_ref0,None)
    kp_ref1, des_ref1 = surf.detectAndCompute(blk_ref1,None)

    if des_target is None or (des_ref0 is None and des_ref1 is None):
        if des_ref0 is None:
            return blk_ref1
        else:
            return blk_ref0

    # Initiate bf matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
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
        chosen_matches = matches0
        kp_ref = kp_ref0
        blk_ref = blk_ref0
    else:
        chosen_matches = matches1
        kp_ref = kp_ref1
        blk_ref = blk_ref1

    dst_pts = np.float32([kp_target[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_ref[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    
    if(len(dst_pts) > 8):
        # calculate homography matrix if there's enough points
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        compensated_block = cv2.warpAffine(blk_ref, M, (blk_ref0.shape[1], blk_ref0.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))

        result = np.zeros_like(compensated_block)
        blk_lun = cv2.resize(blk_lun, (compensated_block.shape[1], compensated_block.shape[0]))

        ret,thresh=cv2.threshold(compensated_block,1,255,0)
        mm,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

        if contours:
            mask_bool = mm.astype(bool)
            result[mask_bool] = compensated_block[mask_bool]
            result[~mask_bool] = blk_lun[~mask_bool]
        else:
            result = compensated_block
    else:
        result = blk_ref

    return result  
