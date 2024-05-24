import cv2
import os
from select_block import select_blocks

def divide_into_blocks(image, block_size=16):
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append((block, (y,x)))
    return blocks

import os

def get_common_files(folderA, folderB):
    files_in_A = set(os.listdir(folderA))
    files_in_B = set(os.listdir(folderB))
    common_files = files_in_A.intersection(files_in_B)
    return common_files


pred_input_path = './solution/output_0'
gt_input_path = './frames'
output_path =  './solution/output_12'
common_files = get_common_files(pred_input_path, gt_input_path)
num_blocks = 13000
for file in common_files:
    compensated_image = cv2.imread(os.path.join(pred_input_path, file), cv2.IMREAD_GRAYSCALE)
    compensated_image[compensated_image == 255] = 0
    gt_image = cv2.imread(os.path.join(gt_input_path, file), cv2.IMREAD_GRAYSCALE)
    compensated_blocks = divide_into_blocks(compensated_image, 16)
    original_blocks = divide_into_blocks(gt_image, 16)
    selected_blocks = select_blocks(compensated_blocks, original_blocks, num_blocks)
    files = int(file[0:3])    
    cv2.imwrite(os.path.join(output_path, f'{files:03}.png'), compensated_image)
    smap_file = open(os.path.join(output_path, f's_{files:03}.txt'),'w')
    for s in selected_blocks:
        smap_file.write(str(s)+'\n')
    smap_file.close()
# for idx in range(1):
#     output_path = r'C:\Users\zzz\Desktop\solution\output_{}'.format(idx)
#     for file in common_files:
#         compensated_image = cv2.imread(os.path.join(pred_input_path, file), cv2.IMREAD_GRAYSCALE)
#         gt_image = cv2.imread(os.path.join(gt_input_path, file), cv2.IMREAD_GRAYSCALE)
#         compensated_blocks = divide_into_blocks(compensated_image, 16)
#         original_blocks = divide_into_blocks(gt_image, 16)
#         selected_blocks = select_blocks(compensated_blocks, original_blocks, num_blocks)
#         files = int(file[0:3])    
#         output_paths = output_path
#         output_paths = os.path.join(output_paths, f's_{files:03}.txt')
#         print(output_paths)
#         smap_file = open(output_paths,'w')
#         for s in selected_blocks:
#             smap_file.write(str(s)+'\n')
#         smap_file.close()