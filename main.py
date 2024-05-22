import numpy as np
import cv2
import os
import argparse
import glob
import pandas as pd
from PIL import Image
import random

def benchmark(so_path, gt_path):

    image_name = ['%03d.png'% i for i in range(129) if i not in [0, 32, 64, 96, 128]]
    txt_name   = ['s_%03d.txt'% i for i in range(129) if i not in [0, 32, 64, 96, 128]]

    so_img_paths = [os.path.join(so_path,name) for name in image_name]
    so_txt_paths = [os.path.join(so_path,name) for name in txt_name]
    gt_img_paths = [os.path.join(gt_path,name) for name in image_name]

    psnr = []
    for so_img_path, so_txt_path, gt_img_path in zip(so_img_paths, so_txt_paths, gt_img_paths):

        s = np.array(Image.open(so_img_path).convert('L'))
        g = np.array(Image.open(gt_img_path).convert('L'))
        f = open(so_txt_path, 'r')

        mask = []
        for line in f.readlines():
            mask.append(int(line.strip('\n')))
        f.close()
        
        mask = np.array(mask).astype(bool)
        assert np.sum(mask) == 13000, 'The number of selection blocks should be 13000'


        s = s.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16)
        g = g.reshape(2160//16, 16, 3840//16, 16).swapaxes(1, 2).reshape(-1, 16, 16)
        
        s = s[mask]
        g = g[mask]
        assert not (s == g).all(), "The prediction should not be the same as the ground truth"

        mse = np.sum((s-g)**2)/s.size
        psnr.append(10*np.log10(255/mse))
    
    psnr = np.array(psnr)
    avg_psnr = np.sum(psnr) / len(psnr)

    return avg_psnr


def divide_frame_to_blocks(image, block_size=16):
    height, width = image.shape[:2]
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append(block)
    return np.array(blocks)

def reconstruct_frame(image, blocks, block_size=16):
    h, w = image.shape
    image = np.zeros(image.shape, dtype=blocks[0].dtype)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return image

def compute_optical_flow(ref_frame, target_frame, method='farneback'):
    """Compute optical flow using different methods."""
    if method == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(ref_frame, target_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    elif method == 'rlof':
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(ref_frame, target_frame, None)
    elif method == 'pyr_lk':
        lk_params = dict(winSize=(21, 21), maxLevel=3)
        flow = cv2.calcOpticalFlowPyrLK(ref_frame, target_frame, None, **lk_params)
    return flow


def warp_image(image, flow):
    """Warp image using the computed flow."""
    h, w = image.shape
    flow_map = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w))).reshape(h, w, 2) + flow
    warped_image = cv2.remap(image, flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)
    return warped_image

def motion_estimate_optical_flow(block, ref_frame0, ref_frame1, flow0, flow1, block_idx, block_size, method='farneback'):
    """Optical flow model with different methods."""
    h, w = ref_frame0.shape
    block_row = (block_idx // (w // block_size)) * block_size
    block_col = (block_idx % (w // block_size)) * block_size

    flow_block0 = flow0[block_row:block_row+block_size, block_col:block_col+block_size]
    flow_block1 = flow1[block_row:block_row+block_size, block_col:block_col+block_size]

    ref_block0 = ref_frame0[block_row:block_row+block_size, block_col:block_col+block_size]
    ref_block1 = ref_frame1[block_row:block_row+block_size, block_col:block_col+block_size]
    compensated_block0 = warp_image(ref_block0, flow_block0)
    compensated_block1 = warp_image(ref_block1, flow_block1)

    mse_block0 = np.mean((block - compensated_block0) ** 2)
    mse_block1 = np.mean((block - compensated_block1) ** 2)

    if mse_block0 < mse_block1:
        return compensated_block0
    else:
        return compensated_block1

def motion_estimate_block_matching(block, ref_block0, ref_block1, method='basic'):
    """Block matching model with different methods."""
    if method == 'basic':
        mse_ref0 = np.mean((block - ref_block0) ** 2)
        mse_ref1 = np.mean((block - ref_block1) ** 2)
    elif method == 'window_search':
        mse_ref0 = np.mean((block - cv2.blur(ref_block0, (5, 5))) ** 2)
        mse_ref1 = np.mean((block - cv2.blur(ref_block1, (5, 5))) ** 2)
    elif method == 'cross_search':
        mse_ref0 = np.mean((block - ref_block0) ** 2) + np.mean((block - ref_block1) ** 2)
        mse_ref1 = mse_ref0  # Simplified for demonstration
    elif method == 'three_step':
        mse_ref0 = np.mean((block - cv2.GaussianBlur(ref_block0, (3, 3), 0)) ** 2)
        mse_ref1 = np.mean((block - cv2.GaussianBlur(ref_block1, (3, 3), 0)) ** 2)
    
    if mse_ref0 < mse_ref1:
        return ref_block0
    else:
        return ref_block1

def motion_estimate_affine_transform(block, ref_block0, ref_block1, method='basic'):
    """Affine transform model with different methods."""
    h, w = block.shape
    points_src = np.array([[0, 0], [w - 1, 0], [0, h - 1]], dtype=np.float32)
    points_dst = points_src + np.random.randint(-2, 3, points_src.shape).astype(np.float32)

    if method == 'basic':
        M0 = cv2.getAffineTransform(points_src, points_dst)
        M1 = cv2.getAffineTransform(points_dst, points_src)
    elif method == 'noisy_points':
        points_dst += np.random.normal(0, 1, points_dst.shape)
        M0 = cv2.getAffineTransform(points_src, points_dst)
        M1 = cv2.getAffineTransform(points_dst, points_src)
    elif method == 'random_points':
        points_dst = np.random.randint(0, min(h, w), points_dst.shape).astype(np.float32)
        M0 = cv2.getAffineTransform(points_src, points_dst)
        M1 = cv2.getAffineTransform(points_dst, points_src)
    elif method == 'different_init':
        points_dst = points_src + np.random.randint(-5, 6, points_src.shape).astype(np.float32)
        M0 = cv2.getAffineTransform(points_src, points_dst)
        M1 = cv2.getAffineTransform(points_dst, points_src)

    warped_block0 = cv2.warpAffine(ref_block0, M0, (w, h))
    warped_block1 = cv2.warpAffine(ref_block1, M1, (w, h))

    mse_block0 = np.mean((block - warped_block0) ** 2)
    mse_block1 = np.mean((block - warped_block1) ** 2)

    if mse_block0 < mse_block1:
        return warped_block0
    else:
        return warped_block1

def motion_estimate_perspective_transform(block, ref_block0, ref_block1, method='basic'):
    """Perspective transform model with different methods."""
    h, w = block.shape
    points_src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    points_dst = points_src + np.random.randint(-2, 3, points_src.shape).astype(np.float32)

    if method == 'basic':
        M0 = cv2.getPerspectiveTransform(points_src, points_dst)
        M1 = cv2.getPerspectiveTransform(points_dst, points_src)
    elif method == 'noisy_points':
        points_dst += np.random.normal(0, 1, points_dst.shape)
        M0 = cv2.getPerspectiveTransform(points_src, points_dst)
        M1 = cv2.getPerspectiveTransform(points_dst, points_src)
    elif method == 'random_points':
        points_dst = np.random.randint(0, min(h, w), points_dst.shape).astype(np.float32)
        M0 = cv2.getPerspectiveTransform(points_src, points_dst)
        M1 = cv2.getPerspectiveTransform(points_dst, points_src)
    elif method == 'different_init':
        points_dst = points_src + np.random.randint(-5, 6, points_src.shape).astype(np.float32)
        M0 = cv2.getPerspectiveTransform(points_src, points_dst)
        M1 = cv2.getPerspectiveTransform(points_dst, points_src)

    warped_block0 = cv2.warpPerspective(ref_block0, M0, (w, h))
    warped_block1 = cv2.warpPerspective(ref_block1, M1, (w, h))

    mse_block0 = np.mean((block - warped_block0) ** 2)
    mse_block1 = np.mean((block - warped_block1) ** 2)

    if mse_block0 < mse_block1:
        return warped_block0
    else:
        return warped_block1

def motion_estimate_homography_transform(block, ref_block0, ref_block1, method='basic'):
    """Homography transform model with different methods."""
    h, w = block.shape
    points_src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    points_dst = points_src + np.random.randint(-2, 3, points_src.shape).astype(np.float32)

    if method == 'basic':
        H0, _ = cv2.findHomography(points_src, points_dst)
        H1, _ = cv2.findHomography(points_dst, points_src)
    elif method == 'noisy_points':
        points_dst += np.random.normal(0, 1, points_dst.shape)
        H0, _ = cv2.findHomography(points_src, points_dst)
        H1, _ = cv2.findHomography(points_dst, points_src)
    elif method == 'random_points':
        points_dst = np.random.randint(0, min(h, w), points_dst.shape)
        H0, _ = cv2.findHomography(points_src, points_dst)
        H1, _ = cv2.findHomography(points_dst, points_src)
    elif method == 'different_init':
        points_dst = points_src + np.random.randint(-5, 6, points_src.shape).astype(np.float32)
        H0, _ = cv2.findHomography(points_src, points_dst)
        H1, _ = cv2.findHomography(points_dst, points_src)

    warped_block0 = cv2.warpPerspective(ref_block0, H0, (w, h))
    warped_block1 = cv2.warpPerspective(ref_block1, H1, (w, h))

    mse_block0 = np.mean((block - warped_block0) ** 2)
    mse_block1 = np.mean((block - warped_block1) ** 2)

    if mse_block0 < mse_block1:
        return warped_block0
    else:
        return warped_block1



def main():
    parser = argparse.ArgumentParser(description='main function of GMC')
    parser.add_argument('--input_path', default='./frames/', help='path to read input frames')
    parser.add_argument('--output_path', default='./output_12/', help='path to put output files')
    parser.add_argument('--csv_file', default='./processing_order.csv', help='processing order CSV file')
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    image_files = glob.glob(os.path.join(args.input_path, '[0-9][0-9][0-9].png'))
    if not image_files:
        print("Cannot find image files from given link.")
        return

    block_size = 16
    num_blocks = 13000
    selected_blocks = [0] * (32400 - num_blocks) + [1] * num_blocks
    # selected_blocks = [1] * num_blocks + [0] * (32400 - num_blocks)

    df = pd.read_csv(args.csv_file).values
    for row in range(len(df)):
        idx, target, ref0, ref1 = df[row]
        if '(X)' in str(target):
            continue

        target, ref0, ref1 = int(target), int(ref0), int(ref1)
        print(f'frame{target}')
        target_img_path = os.path.join(args.input_path, f'{target:03}.png')
        ref0_img_path = os.path.join(args.input_path, f'{ref0:03}.png')
        ref1_img_path = os.path.join(args.input_path, f'{ref1:03}.png')

        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        ref0_img = cv2.imread(ref0_img_path, cv2.IMREAD_GRAYSCALE)
        ref1_img = cv2.imread(ref1_img_path, cv2.IMREAD_GRAYSCALE)
        
        target_block = divide_frame_to_blocks(target_img)
        ref0_block = divide_frame_to_blocks(ref0_img)
        ref1_block = divide_frame_to_blocks(ref1_img)
        
        # Compute optical flow using different methods
        # flow0_farneback = compute_optical_flow(ref0_img, target_img, method='farneback')
        # flow1_farneback = compute_optical_flow(ref1_img, target_img, method='farneback')
        # flow0_rlof = compute_optical_flow(ref0_img, target_img, method='rlof')
        # flow1_rlof = compute_optical_flow(ref1_img, target_img, method='rlof')
        # flow0_pyr_lk = compute_optical_flow(ref0_img, target_img, method='pyr_lk')
        # flow1_pyr_lk = compute_optical_flow(ref1_img, target_img, method='pyr_lk')

        
        compensated_blocks = []
        for idx, t_block in enumerate(target_block):
            # model = random.choice([
            #         'block_matching_basic', 'block_matching_window', 'block_matching_cross', 'block_matching_three_step',
            #         'optical_flow_farneback', 'optical_flow_rlof', 'optical_flow_pyr_lk',
            #         'affine_basic', 'affine_noisy_points', 'affine_random_points', 'affine_different_init',
            #         'perspective_basic', 'perspective_noisy_points', 'perspective_random_points', 'perspective_different_init',
            #         'homography_basic', 'homography_noisy_points', 'homography_random_points', 'homography_different_init'
            #         ])
            # if model.startswith('block_matching'):
            #     method = model.split('_')[2]
            #     best_compensated_block = motion_estimate_block_matching(t_block, ref0_block[idx], ref1_block[idx], method=method)
            # elif model.startswith('optical_flow'):
            #     method = model.split('_')[2]
            #     if method == 'farneback':
            #         flow0, flow1 = flow0_farneback, flow1_farneback
            #     elif method == 'rlof':
            #         flow0, flow1 = flow0_rlof, flow1_rlof
            #     elif method == 'pyr_lk':
            #         flow0, flow1 = flow0_pyr_lk, flow1_pyr_lk
            #     best_compensated_block = motion_estimate_optical_flow(t_block, ref0_img, ref1_img, flow0, flow1, idx, block_size, method=method)
            # elif model.startswith('affine'):
            #     method = model.split('_')[1]
            #     best_compensated_block = motion_estimate_affine_transform(t_block, ref0_block[idx], ref1_block[idx], method=method)
            # elif model.startswith('perspective'):
            #     method = model.split('_')[1]
            #     best_compensated_block = motion_estimate_perspective_transform(t_block, ref0_block[idx], ref1_block[idx], method=method)
            # elif model.startswith('homography'):
            #     method = model.split('_')[1]
            #     best_compensated_block = motion_estimate_homography_transform(t_block, ref0_block[idx], ref1_block[idx], method=method)
                
            # compensated_blocks.append(best_compensated_block)

            # best_ref_block = motion_estimate_block_matching(t_block, ref0_block[idx], ref1_block[idx], method='window_search')
            # best_ref_block = motion_estimate_block_matching(t_block, ref0_block[idx], ref1_block[idx], method='cross_search')
            # best_ref_block = motion_estimate_block_matching(t_block, ref0_block[idx], ref1_block[idx], method='three_step')
            # best_ref_block = motion_estimate_homography_transform(t_block, ref0_block[idx], ref1_block[idx], method='basic')
            # best_ref_block = motion_estimate_homography_transform(t_block, ref0_block[idx], ref1_block[idx], method='noisy_points')
            best_ref_block = motion_estimate_homography_transform(t_block, ref0_block[idx], ref1_block[idx], method='random_points')
            # best_ref_block = motion_estimate_homography_transform(t_block, ref0_block[idx], ref1_block[idx], method='basic')
            compensated_blocks.append(best_ref_block)
        
    
        compensated_img = reconstruct_frame(target_img, compensated_blocks, block_size)
        cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_img)
        
        # block_matching_algo = 'full_search'
        # motion_vectors_ref0_to_target, motion_vectors_ref1_to_target = motion_estimation(ref0_img, ref1_img, target_img, block_matching_algo)
        # motion_vectors = (motion_vectors_ref0_to_target, motion_vectors_ref1_to_target)
        # compensated_img = global_motion_compensation(ref0_img, ref1_img, motion_vectors, block_size=block_size)
        # cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_img)
        
        # src_pts, dst_pts = brisk_feature_matching(target_img, ref0_img, ref1_img)
        # src_pts, dst_pts = sift_feature_matching(target_img, ref0_img, ref1_img)
        
        # compensated_img = apply_global_motion_compensation_Affine(target_img, ref0_img, src_pts, dst_pts)
        # cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_img)
        
        # compensated_img = apply_global_motion_compensation_Perspective(target_img, ref0_img, src_pts, dst_pts)
        # cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_img)
        
        # compensated_img = apply_dense_optical_flow_compensation(target_img, ref0_img, ref1_img)
        # cv2.imwrite(os.path.join(args.output_path, f'{target:03}.png'), compensated_img)
        
        output_path = os.path.join(args.output_path, f's_{target:03}.txt')
        smap_file = open(output_path,'w')
        for s in selected_blocks:
            smap_file.write(str(s)+'\n')
    smap_file.close()
    
    so_path = args.output_path
    gt_path = args.input_path
     
    score = benchmark(so_path, gt_path)

    print('PSNR: %.5f\n'%(score))


if __name__ == '__main__':
    main()
    

# 你現在有129個original frame，用for讀取reference frame 0 跟 reference frame 1 跟 target frame，每種frame的size都是 3840 * 2160，並將 previous frame 0、next frame 1、target frame各自切割成32400個block，每個block 大小為 16 * 16，目標是要以這些block作為input，然後對每個block套用他適合的motion estimate跟motion compensation model (可以是affin/perspectiove/projection/machine learning...whatever)，每個block可以使用不同model，最後再把這些block組成compensated image，並與original image去計算PSNR，PSNR的function已經給了