import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd

from gmc import feature_matching

def divide_into_128_blocks(image):
    h, w = image.shape
    block_h, block_w = h // 4, w // 8  # 8行，每行16個區塊
    blocks = []
    for y in range(0, h, block_h):
        for x in range(0, w, block_w):
            block = image[y:y+block_h, x:x+block_w]
            blocks.append((block, (y, x)))
    return blocks

def divide_into_blocks(image, block_size):
    # Divide an image into non-overlapping blocks of the specified size.
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

def reconstruct_image_from_blocks(blocks, image_shape):
    compensated_image = np.zeros(image_shape, dtype=np.uint8)
    for block, (y, x) in blocks:
        block_h, block_w = block.shape[:2]
        compensated_image[y:y+block_h, x:x+block_w] = block

    return compensated_image

def select_blocks(blocks, num_blocks):
    selected_blocks = []
    
    # TODO: (For test ONLY) select first 13000 blocks
    for idx in range(len(blocks)):
        if idx < num_blocks:
            selected_blocks.append(1)
        else:
            selected_blocks.append(0)

    return selected_blocks


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

    block_size = 16
    num_blocks = 13000

    df = pd.read_csv(args.csv_file)

    for idx, row in df.iterrows():
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

        # # Step 2: Divide the image into 16x16 blocks
        # target_blocks = divide_into_blocks(target_img, block_size)
        # ref0_blocks = divide_into_blocks(ref0_img, block_size)
        # ref1_blocks = divide_into_blocks(ref1_img, block_size)

        target_blocks = divide_into_128_blocks(target_img)
        ref0_blocks = divide_into_128_blocks(ref0_img)
        ref1_blocks = divide_into_128_blocks(ref1_img)

        # Step 3:
        # TODO: apply motion model (gmc.py)
        flow = cv2.calcOpticalFlowFarneback(ref0_img, target_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        compensated_blocks = []
        for ((blk_target, (y,x)), (blk_ref0, (a,b)), (blk_ref1, (c,d))) in zip(target_blocks, ref0_blocks, ref1_blocks):
            compensated_block = feature_matching(blk_target, blk_ref0, blk_ref1, flow[y:y+blk_target.shape[0], x:x+blk_target.shape[1]])
            compensated_blocks.append((compensated_block, (y,x)))

        # 根據補償後的block重建整張圖片
        
        compensated_image = reconstruct_image_from_blocks(compensated_blocks, target_img.shape)
        cv2.imwrite(os.path.join(args.output_path, 'compensated_image.png'), compensated_image)
        # break

        # Step 4: Select 13,000 blocks
        selected_blocks = select_blocks(target_blocks, num_blocks)

        # Step 5: output selection map: s_xxx.txt
        output_path = os.path.join(args.output_path, f's_{target:03}.txt')
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()


if __name__ == '__main__':
    main()

