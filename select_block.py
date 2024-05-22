import numpy as np

def calculate_psnr(block, reference_block):
    mse = np.mean((block - reference_block) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0  # Assuming the pixel values range from 0 to 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def select_blocks(blocks, reference_blocks, num_blocks):
    '''
    select rules:
    1. black pixels the less the better
    2. psnr the higher the better
    '''
    black_pixel_counts = np.zeros(len(blocks))
    psnr_scores = np.zeros(len(blocks))

    for idx, (block,(y,x)) in enumerate(blocks):
        # Count the number of black pixels
        black_pixel_counts[idx] = np.sum(block == 0)

        (ref_block,(a,b)) = reference_blocks[idx]
        psnr_scores[idx] = calculate_psnr(block, ref_block)

    # Normalize 
    black_pixel_ranks = np.argsort(black_pixel_counts)
    psnr_ranks = np.argsort(-psnr_scores)

    combined_ranks = black_pixel_ranks + psnr_ranks
    sorted_indices = np.argsort(combined_ranks  )
    output_list = [0] * len(blocks)

    for i in sorted_indices[:num_blocks]:
        output_list[i] = 1

    return output_list