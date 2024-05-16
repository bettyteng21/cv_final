import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd

from gmc import block_matching, apply_motion_compensation

def divide_into_blocks(image, block_size):
    # Divide an image into non-overlapping blocks of the specified size.
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

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

        # Step 2: Divide the image into 16x16 blocks
        target_blocks = divide_into_blocks(target_img, block_size)
        ref0_blocks = divide_into_blocks(ref0_img, block_size)
        ref1_blocks = divide_into_blocks(ref1_img, block_size)

        # Step 3:
        # TODO: apply motion model (gmc.py)

        
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

