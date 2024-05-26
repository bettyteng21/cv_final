import cv2
import os

def post_process(target_image, target_frame_idx, gt_path):
    frame_list = []

    for i in range(target_frame_idx-2, target_frame_idx+3):
        if i == target_frame_idx:
            frame_list.append(target_image)
        else:
            img_path = os.path.join(gt_path, f'{i:03}.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            frame_list.append(img)

    if (target_frame_idx-2)>=0 and (target_frame_idx+3)<=128:
        result = cv2.fastNlMeansDenoisingMulti(frame_list, 2, 5, None, 3, 7, 21)
    else:
        result = cv2.fastNlMeansDenoising(img, None, 3, 7, 21)

    return result
