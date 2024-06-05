import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd

from gmc import motion_compensate_for_nbb
from select_block import select_blocks

def divide_into_blocks(image, block_size):
    # 將圖片切成以block_size為單位的blocks
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:min(y+block_size, height), x:min(x+block_size, width)]
            blocks.append((block, (y,x)))
    return blocks


def reconstruct_image_from_blocks(blocks, image_shape):
    # 利用blocks裡的座標(y,x)，貼回去一張空白的圖，最後輸出一張合成圖
    compensated_image = np.full(image_shape, 80, dtype=np.uint8)
    for block, (y, x) in blocks:
        block_h, block_w = block.shape[:2]
        
        if y + block_h > image_shape[0] or x + block_w > image_shape[1]:
            # 當超出圖片範圍時，resize成可以fit進去圖片的大小
            block = cv2.resize(block, (min(block_w, image_shape[1]-x), min(block_h, image_shape[0]-y)), interpolation=cv2.INTER_CUBIC)

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
    
    if not os.path.exists('model_map'):
        os.makedirs('model_map')

    num_blocks = 13000
    psnr_list = []

    df = pd.read_csv(args.csv_file)

    for idx, row in df.iterrows():

        target = row['Target Picture']
        ref0 = row['Reference Pic0']
        ref1 = row['Reference Pic1']

        if '(X)' in str(target):
            # skip the ones that are labeled (X)
            continue
        
        # Step 1: Load img and files
        target, ref0, ref1 = int(target), int(ref0), int(ref1)
        print('\nCurrent processing order '+str(idx)+', target frame '+str(target))

        model_map_path = os.path.join('model_map', f'm_{target:03}.txt')
        if os.path.isfile(model_map_path):
            open(model_map_path, 'w').close()
        
        target_img_path = os.path.join(args.input_path, f'{target:03}.png')
        ref0_img_path = os.path.join(args.input_path, f'{ref0:03}.png')
        ref1_img_path = os.path.join(args.input_path, f'{ref1:03}.png')

        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        ref0_img = cv2.imread(ref0_img_path, cv2.IMREAD_GRAYSCALE)
        ref1_img = cv2.imread(ref1_img_path, cv2.IMREAD_GRAYSCALE)

        # Step 2: global motion compensation (gmc.py), get a list of compensated blocks
        compensated_blocks = []
        blocks = divide_into_blocks(target_img, 400)
        temp_blocks = motion_compensate_for_nbb(blocks, ref0_img, ref1_img, target)
        compensated_blocks.extend(temp_blocks)

        # Step 3: construct the compensated image
        compensated_image = reconstruct_image_from_blocks(compensated_blocks, target_img.shape)
        cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_image)

        # Step 4: Select 13,000 blocks of the top psnr (select_block.py)
        compensated_blocks = divide_into_blocks(compensated_image, 16)
        original_blocks = divide_into_blocks(target_img, 16)
        selected_blocks = select_blocks(compensated_blocks, original_blocks, num_blocks)
        
        # Step 5: eval for printing current psnr
        mask = np.array(selected_blocks).astype(bool)
        assert np.sum(mask) == 13000, 'The number of selection blocks should be 13000'
        s = compensated_image.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16).astype(int)
        g = target_img.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16).astype(int)
        s = s[mask]
        g = g[mask]
        assert not (s == g).all(), "The prediction should not be the same as the ground truth"
        mse = np.sum((s-g)**2)/s.size
        psnr_curr = 10*np.log10((255**2)/mse)
        psnr_list.append(psnr_curr)
        print('Current psnr= '+str(psnr_curr))

        # Step 6: output selection map: s_xxx.txt
        output_path = os.path.join(args.output_path, f's_{target:03}.txt')
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
        smap_file.close()

    psnr_list = np.array(psnr_list)
    avg_psnr = np.mean(psnr_list)
    print('Avg psnr: '+str(avg_psnr))

if __name__ == '__main__':
    main()
