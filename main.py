import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd

from gmc import motion_compensate_for_nbb
# from detect_block import load_yolo_model, detect_objects, find_corresponding_blocks, retrieve_bounding_box_image
from select_block import select_blocks

# # Divide image into blocks based on detected objects,
# # return blocks and the masked areas without blocks(1 for no block, 0 for has block)
# def divide_into_object_based_blocks(image, boxes):
#     blocks = []
#     mask = np.zeros(image.shape, dtype=np.uint8)
#     for (x, y, w, h) in boxes:
#         block = image[y:y+h, x:x+w]
#         blocks.append((block, (y, x)))
#         mask[y:y+h, x:x+w] = 255
#     return blocks, cv2.bitwise_not(mask)

# Divide an image into non-overlapping blocks of the specified size.
def divide_into_blocks(image, block_size):
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:min(y+block_size, height), x:min(x+block_size, width)]
            blocks.append((block, (y,x)))
    return blocks

# Paste the blocks back to an image
def reconstruct_image_from_blocks(blocks, image_shape):
    compensated_image = np.full(image_shape, 80, dtype=np.uint8)
    for block, (y, x) in blocks:
        block_h, block_w = block.shape[:2]
        # Ensure the block fits within the image dimensions
        if y + block_h > image_shape[0] or x + block_w > image_shape[1]:
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

    # Load models
    # net, output_layers = load_yolo_model()

    for idx, row in df.iterrows():

        target = row['Target Picture']
        ref0 = row['Reference Pic0']
        ref1 = row['Reference Pic1']

        if '(X)' in str(target):
            # skip the ones that are labeled (X)
            continue
        
        # Step: Load img
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
        
        # # Step: Detect objects in the foreground (detect_block.py)
        # ref0_boxes = detect_objects(ref0_img, net, output_layers)
        # ref1_boxes = detect_objects(ref1_img, net, output_layers)
        # target_boxes = detect_objects(target_img, net, output_layers)

        # target_blocks, target_mask = divide_into_object_based_blocks(target_img, target_boxes)
        # ref0_blocks, ref0_mask = divide_into_object_based_blocks(ref0_img, ref0_boxes)
        # ref1_blocks, ref1_mask = divide_into_object_based_blocks(ref1_img, ref1_boxes)

        # target_boxes_img = []
        # ref0_boxes_img = []
        # ref1_boxes_img = []
        # for i in range(len(target_blocks)):
        #     img = retrieve_bounding_box_image(target_blocks[i])
        #     if img.size != 0:
        #         target_boxes_img.append(img)

        # for i in range(len(ref0_blocks)):
        #     img = retrieve_bounding_box_image(ref0_blocks[i])
        #     if img.size != 0:
        #         ref0_boxes_img.append(img)

        # for i in range(len(ref1_blocks)):
        #     img = retrieve_bounding_box_image(ref1_blocks[i])
        #     if img.size != 0:
        #         ref1_boxes_img.append(img)

        # # 找出target obj對應的ref0 obj & ref1 obj，並記錄他們對應的index
        # ref0_mapping = find_corresponding_blocks(target_boxes_img, ref0_boxes_img, threshold=10)
        # ref1_mapping = find_corresponding_blocks(target_boxes_img, ref1_boxes_img, threshold=10)

        # Step: apply motion model (gmc.py)
        compensated_blocks = []

        # Step-1: gmc for background (non-bounding-box)
        blocks = divide_into_blocks(target_img, 400)
        temp_blocks = motion_compensate_for_nbb(blocks, ref0_img, ref1_img, target)
        compensated_blocks.extend(temp_blocks)

        # # Step-2: gmc for detected objects
        # for ((blk_target, (y,x)),idx_ref0, idx_ref1) in zip(target_blocks, ref0_mapping, ref1_mapping):
        #     (blk_ref0, (y0,x0)),(blk_ref1, (y1,x1)) = ref0_blocks[idx_ref0], ref1_blocks[idx_ref1]
        #     if idx_ref0 == -1 and idx_ref1 ==-1:
        #         continue # no corresponding object
        #     elif idx_ref0 == -1:
        #         blk_ref0 = np.full_like(blk_ref0, 80)
        #     elif idx_ref1 == -1:
        #         blk_ref1 = np.full_like(blk_ref1, 80)

        #     compensated_block = feature_matching(blk_target, blk_ref0, blk_ref1, idx_ref0, idx_ref1)
        #     compensated_blocks.append((compensated_block, (y,x)))
        
        compensated_image = reconstruct_image_from_blocks(compensated_blocks, target_img.shape)

        cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_image)

        # Step: Select 13,000 blocks (select_block.py)
        compensated_blocks = divide_into_blocks(compensated_image, 16)
        original_blocks = divide_into_blocks(target_img, 16)
        selected_blocks = select_blocks(compensated_blocks, original_blocks, num_blocks)
        
        # ##############################################
        # eval
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
        
        # ###############################################

        # Step: output selection map: s_xxx.txt
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
