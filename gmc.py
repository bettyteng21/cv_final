import numpy as np
import cv2
import os

def write_mmap(image, block_size, input_value, input_idx):
    # 將傳入的model值寫入model map
    height, width = image.shape[:2]
    model_map_path = os.path.join('model_map', f'm_{input_idx:03}.txt')
    f = open(model_map_path, 'a')
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            f.write(str(input_value)+'\n')
    f.close()


def motion_compensate_for_nbb(blocks, ref0_image, ref1_image, target_idx):
    compensated_blocks = []
    for block,(y,x) in blocks:
        # 從target img block的座標(y,x)取出相同位置的ref0 & ref1 image block
        ref0_block = ref0_image[y:y+block.shape[0], x:x+block.shape[1]]
        ref1_block = ref1_image[y:y+block.shape[0], x:x+block.shape[1]]
        
        # Apply feature matching for motion compensation
        compensated_block = feature_matching_for_nbb(block, ref0_block, ref1_block, target_idx)
        compensated_blocks.append((compensated_block, (y, x)))

    return compensated_blocks


def feature_matching_for_nbb(blk_target, blk_ref0, blk_ref1, target_idx):
    '''
    真正主要做matching的fuction
    @param target_idx: 為了後面write_mmap知道要開啟哪個檔案

    model說明:
    model 1: 使用ref0 block, 經過affine/projection transformation得到結果
    model 2: 使用ref1 block, 經過affine/projection transformation得到結果
    model 3: 直接使用ref0 block
    model 4: 直接使用ref1 block
    '''

    # Step 1: 用surf找出target, ref0, ref1各自的特徵
    surf = cv2.xfeatures2d.SURF_create(300)
    kp_target, des_target = surf.detectAndCompute(blk_target,None)
    kp_ref0, des_ref0 = surf.detectAndCompute(blk_ref0,None)
    kp_ref1, des_ref1 = surf.detectAndCompute(blk_ref1,None)

    if des_target is None or (des_ref0 is None and des_ref1 is None):
        if des_ref0 is None:
            write_mmap(blk_ref1, 16, 4, target_idx) # write model 4
            return blk_ref1
        else:
            write_mmap(blk_ref0, 16, 3, target_idx) # write model 3
            return blk_ref0


    # Step 2: 用bf做matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches0, matches1 = [],[]
    if des_ref0 is not None:
        matches0 = bf.match(des_target, des_ref0) 
        matches0 = sorted(matches0, key=lambda x: x.distance)

    if des_ref1 is not None:
        matches1 = bf.match(des_target, des_ref1) 
        matches1 = sorted(matches1, key=lambda x: x.distance)


    # Step 3: 將剛剛match的距離加總，看是ref0還是ref1的距離較小，較小的那個代表跟target比較相似
    if matches0 is not None:
        total_distance0 = sum([m.distance for m in matches0])
    if matches1 is not None:
        total_distance1 = sum([m.distance for m in matches1])

    if total_distance0 < total_distance1:
        chosen_matches = matches0
        kp_ref = kp_ref0
        blk_ref = blk_ref0
        ref_select = 0
    else:
        chosen_matches = matches1
        kp_ref = kp_ref1
        blk_ref = blk_ref1
        ref_select = 1


    # Step 4: 取出比較相似的那張ref的點
    dst_pts = np.float32([kp_target[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp_ref[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 1, 2)
    

    # Step 5: 計算projection
    if len(dst_pts) > 8:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        compensated_block = cv2.warpPerspective(blk_ref, M, (blk_ref0.shape[1], blk_ref0.shape[0]), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)

        if ref_select == 0:
            write_mmap(compensated_block, 16, 1, target_idx) # write model 1
        else: # ref_select==1
            write_mmap(compensated_block, 16, 2, target_idx) # write model 2
        
    else:
        # 當點不夠的時候就不計算，直接用model 3 or 4
        compensated_block = blk_ref

        if ref_select == 0:
            write_mmap(compensated_block, 16, 3, target_idx) # write model 3
        else: # ref_select==1
            write_mmap(compensated_block, 16, 4, target_idx) # write model 4

    return compensated_block