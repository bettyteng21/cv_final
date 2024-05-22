import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd

from gmc import feature_matching, motion_compensate_for_nbb
# from segment import load_deeplab_model, segment_image, extract_foreground
from detect_block import draw_boxes, load_yolo_model, detect_objects, find_corresponding_blocks, retrieve_bounding_box_image
from select_block import calculate_psnr, select_blocks

# Divide image into blocks based on detected objects,
# return blocks and the masked areas without blocks(1 for no block, 0 for has block)
def divide_into_object_based_blocks(image, boxes):
    blocks = []
    mask = np.zeros(image.shape, dtype=np.uint8)
    for (x, y, w, h) in boxes:
        block = image[y:y+h, x:x+w]
        blocks.append((block, (y, x)))

        mask[y:y+h, x:x+w] = 255

    return blocks, cv2.bitwise_not(mask)

# Divide an image into non-overlapping blocks of the specified size.
def divide_into_blocks(image, block_size):
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append((block, (y,x)))
    return blocks

# Paste the blocks back to an image
def reconstruct_image_from_blocks(blocks, image_shape):
    compensated_image = np.zeros(image_shape, dtype=np.uint8)
    for block, (y, x) in blocks:
        block_h, block_w = block.shape[:2]
        # Ensure the block fits within the image dimensions
        if y + block_h > image_shape[0] or x + block_w > image_shape[1]:
            block = cv2.resize(block, (min(block_w, image_shape[1]-x), min(block_h, image_shape[0]-y)))
        compensated_image[y:y+block_h, x:x+block_w] = block

    return compensated_image


def main():
    parser = argparse.ArgumentParser(description='main function of GMC')
    parser.add_argument('--input_path', default='./frames/', help='path to read input frames')
    parser.add_argument('--output_path', default='./output/', help='path to put output files')
    parser.add_argument('--csv_file', default='./processing_order.csv', help='processing order CSV file')
    args = parser.parse_args()

    image_files = glob.glob(os.path.join(args.input_path, '[0-9][0-9][0-9].png'))
    if not image_files:
        print("Cannot find image files from given link.")
        return
    
    if not os.path.exists(os.path.join(args.output_path, 'compensate')): 
        # make folder to put compensated image
        os.makedirs(os.path.join(args.output_path, 'compensate'))

    if not os.path.exists(os.path.join(args.output_path, 'smap_txt')): 
        # make folder to put selected 13000 blocks
        os.makedirs(os.path.join(args.output_path, 'smap_txt'))

    block_size = 128
    num_blocks = 13000
    psnr_list = []

    df = pd.read_csv(args.csv_file)

    # Load models
    net, output_layers = load_yolo_model()
    # segmentation_model = load_deeplab_model()

    for idx, row in df.iterrows():
        print('Current processing order: '+str(idx))

        target = row['Target Picture']
        ref0 = row['Reference Pic0']
        ref1 = row['Reference Pic1']

        if '(X)' in str(target):
            # skip the ones that are labeled (X)
            continue
        
        # Step 1: Load img
        target, ref0, ref1 = int(target), int(ref0), int(ref1)
        target_img_path = os.path.join(args.input_path, f'{target:03}.png')
        ref0_img_path = os.path.join(args.input_path, f'{ref0:03}.png')
        ref1_img_path = os.path.join(args.input_path, f'{ref1:03}.png')

        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        ref0_img = cv2.imread(ref0_img_path, cv2.IMREAD_GRAYSCALE)
        ref1_img = cv2.imread(ref1_img_path, cv2.IMREAD_GRAYSCALE)
        
        # # Step 2: Segment the images to get foreground masks
        # target_mask = segment_image(target_img, segmentation_model)
        # ref0_mask = segment_image(ref0_img, segmentation_model)

        # # Step 3: Extract foreground
        # target_foreground = extract_foreground(target_img, target_mask)
        # ref0_foreground = extract_foreground(ref0_img, ref0_mask)

        # # Draw and display the detected blocks
        # ref0_foreground = cv2.resize(ref0_foreground, (target_foreground.shape[1]//2, target_foreground.shape[0]//2))
        # cv2.imshow("target_foreground", ref0_foreground)
        # cv2.waitKey(0)  # Press any key to continue

        # Step 4: Detect objects in the foreground
        ref0_boxes = detect_objects(ref0_img, net, output_layers)
        ref1_boxes = detect_objects(ref1_img, net, output_layers)
        target_boxes = detect_objects(target_img, net, output_layers)

        target_blocks, target_mask = divide_into_object_based_blocks(target_img, target_boxes)
        ref0_blocks, ref0_mask = divide_into_object_based_blocks(ref0_img, ref0_boxes)
        ref1_blocks, ref1_mask = divide_into_object_based_blocks(ref1_img, ref1_boxes)

        target_boxes_img = []
        ref0_boxes_img = []
        ref1_boxes_img = []
        for i in range(len(target_blocks)):
            img = retrieve_bounding_box_image(target_blocks[i])
            if img.size != 0:
                target_boxes_img.append(img)
                # cv2.imwrite('./output/target/'+str(i)+'.png', img)

        for i in range(len(ref0_blocks)):
            img = retrieve_bounding_box_image(ref0_blocks[i])
            if img.size != 0:
                ref0_boxes_img.append(img)
                # cv2.imwrite('./output/ref0/'+str(i)+'.png', img)

        for i in range(len(ref1_blocks)):
            img = retrieve_bounding_box_image(ref1_blocks[i])
            if img.size != 0:
                ref1_boxes_img.append(img)
                # cv2.imwrite('./output/ref1/'+str(i)+'.png', img)

        ref0_mapping = find_corresponding_blocks(target_boxes_img, ref0_boxes_img, threshold=10)
        ref1_mapping = find_corresponding_blocks(target_boxes_img, ref1_boxes_img, threshold=10)

        # Output the mapping
        with open(f"{args.output_path}/mapping.txt", "w") as f:
            for i, (r0, r1) in enumerate(zip(ref0_mapping, ref1_mapping)):
                f.write(f"Target object {i}: ref0 block {r0}, ref1 block {r1}\n")
        
        # Draw and display the detected blocks
        image_with_boxes = draw_boxes(target_img.copy(), target_boxes)
        image_with_boxes = cv2.resize(image_with_boxes, (image_with_boxes.shape[1]//2, image_with_boxes.shape[0]//2))
        # cv2.imwrite(os.path.join(args.output_path, 'detected_blocks.png'), image_with_boxes)

        # Step 5: apply motion model (gmc.py)
        compensated_blocks = []

        # gmc for background (non-bounding-box)
        blocks = divide_into_blocks(target_img, block_size)
        nbb_blocks = [(blk,coord) for blk,coord in blocks if target_mask[coord[0]:coord[0]+blk.shape[0], coord[1]:coord[1]+blk.shape[1]].sum() > 0]
        temp_blocks = motion_compensate_for_nbb(nbb_blocks, ref0_img, ref1_img)
        compensated_blocks.extend(temp_blocks)

        # gmc for detected objects
        for ((blk_target, (y,x)),idx_ref0, idx_ref1) in zip(target_blocks, ref0_mapping, ref1_mapping):
            (blk_ref0, (y0,x0)),(blk_ref1, (y1,x1)) = ref0_blocks[idx_ref0], ref1_blocks[idx_ref1]
            if idx_ref0 == -1 and idx_ref1 ==-1:
                continue
            elif idx_ref0 == -1:
                blk_ref0 = np.zeros_like(blk_ref0)
            elif idx_ref1 == -1:
                blk_ref1 = np.zeros_like(blk_ref1)

            compensated_block = feature_matching(blk_target, blk_ref0, blk_ref1, idx_ref0, idx_ref1)
            compensated_blocks.append((compensated_block, (y,x)))
        
        compensated_image = reconstruct_image_from_blocks(compensated_blocks, target_img.shape)

        cv2.imwrite(os.path.join(args.output_path, 'compensate/'+str(idx)+'.png'), compensated_image)

        # Step 6: Select 13,000 blocks
        compensated_blocks = divide_into_blocks(compensated_image, 16)
        original_blocks = divide_into_blocks(target_img, 16)
        selected_blocks = select_blocks(compensated_blocks, original_blocks, num_blocks)

        # Step 7: output selection map: s_xxx.txt
        output_path = os.path.join(args.output_path, f'smap_txt/s_{target:03}.txt')
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()

        psnr = calculate_psnr(compensated_image, target_img)
        psnr_list.append(psnr)
    
    # psnr for every FULL compensated image
    psnr_path = os.path.join(args.output_path, f'psnr.txt')
    psnr_file = open(psnr_path,'w')
    for s in psnr_list:
        psnr_file.write(str(s)+'\n')
    psnr_file.close()

if __name__ == '__main__':
    main()

