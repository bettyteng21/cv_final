import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd
from gmc import *
from select_block import select_blocks
from eval import benchmark

def divide_into_blocks(image, block_size=16):
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append((block, (y,x)))
    return blocks


def main():
    parser = argparse.ArgumentParser(description='main function of GMC')
    parser.add_argument('--input_path', default='./frames/', help='path to read input frames')
    parser.add_argument('--output_path', default='./1616/', help='path to put output files')
    parser.add_argument('--csv_file', default='./processing_order.csv', help='processing order CSV file')
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    image_files = glob.glob(os.path.join(args.input_path, '[0-9][0-9][0-9].png'))
    if not image_files:
        print("Cannot find image files from given link.")
        return
    
    block_size = 16
    search_range = 16
    num_blocks = 13000
    frame_list = ['8', '12', '16', '20', '24', 
                  '28', '36', '40', '44', '48', 
                  '52', '56', '60', '72', '80', 
                  '88', '104', '112']
    df = pd.read_csv(args.csv_file)
    
    for idx, row in df.iterrows():
        target = row['Target Picture']
        ref0 = row['Reference Pic0']
        ref1 = row['Reference Pic1']
        if '(X)' in str(target):
            continue
        
        target_str, ref0_str, ref1_str = target, ref0, ref1
        target, ref0, ref1 = int(target), int(ref0), int(ref1)
        target_img_path = os.path.join(args.input_path, f'{target:03}.png')
        ref0_img_path = os.path.join(args.input_path, f'{ref0:03}.png')
        ref1_img_path = os.path.join(args.input_path, f'{ref1:03}.png')

        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        ref0_img = cv2.imread(ref0_img_path, cv2.IMREAD_GRAYSCALE)
        ref1_img = cv2.imread(ref1_img_path, cv2.IMREAD_GRAYSCALE)
        print(target_str)
        
        motion_vectors_ref0_to_target, motion_vectors_ref1_to_target = motion_estimation(ref0_img, ref1_img, target_img, block_size=block_size, search_range=search_range)
        motion_vectors = (motion_vectors_ref0_to_target, motion_vectors_ref1_to_target)
        compensated_img = global_motion_compensation(ref0_img, ref1_img, motion_vectors, block_size=block_size)

        cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_img)
        
        compensated_blocks = divide_into_blocks(compensated_img, 16)
        original_blocks = divide_into_blocks(target_img, 16)
        selected_blocks = select_blocks(compensated_blocks, original_blocks, num_blocks)
        
        output_path = os.path.join(args.output_path, f's_{target:03}.txt')
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()

    score = benchmark(args.output_path, args.input_path)
    print('PSNR: %.5f\n'%(score))

if __name__ == '__main__':
    main()