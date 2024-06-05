import os
import argparse
import numpy as np
from PIL import Image
import shutil

def convert_to_yuv(so_path, gt_path, png_dir, output_file, num_frames):
    
    # genenerate a file with and only all .png
    gt_image_name, so_image_name = [], []
    for i in range(129):
        if i in [0, 32, 64, 96, 128]:
            gt_image_name.append(f'{i:03}.png')
        else:
            so_image_name.append(f'{i:03}.png')
    gt_img_paths = [os.path.join(gt_path, name) for name in gt_image_name]
    so_img_paths = [os.path.join(so_path, name) for name in so_image_name]
    
    os.makedirs(png_dir, exist_ok=True)

    for img_path in gt_img_paths + so_img_paths:
        shutil.copy(img_path, png_dir)

    # use the generate file to generate .yuv
    with open(output_file, "wb") as f_y:
        for frame_num in range(num_frames):
            # Load the image
            image_path = os.path.join(png_dir, f'{frame_num:03d}.png')
            with Image.open(image_path) as img:
                # Ensure the image is in grayscale mode
                img = img.convert('L')
                y_plane = np.array(img, dtype=np.uint8)

                # Write the Y plane to the YUV file
                f_y.write(y_plane.tobytes())

def main():
    parser = argparse.ArgumentParser(description='Convert PNG images to a YUV file.')
    parser.add_argument('--so_path', type=str, required=True, help='Path to the source images directory')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth images directory')
    parser.add_argument('--png_dir', type=str, required=True, help='Directory to store temporary PNG images')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Path to the output YUV file')
    parser.add_argument('-n', '--num_frames', type=int, required=True, help='Number of frames to convert')

    args = parser.parse_args()

    convert_to_yuv(args.so_path, args.gt_path, args.png_dir, args.output_file, args.num_frames)

if __name__ == "__main__":
    main()
