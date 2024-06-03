import numpy as np
from multiprocessing import Pool

def block_matching(ref_frame, target_block, y, x, block_size, search_range):
    def match(ref_frame, y, x, block_size, search_range):
        ref_blocks = []
        ref_blocks_dst = []
        height, width = ref_frame.shape
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                y0, x0 = y + dy, x + dx
                if 0 <= y0 < height - block_size and 0 <= x0 < width - block_size:
                    ref_block = ref_frame[y0:y0 + block_size, x0:x0 + block_size].flatten()
                    ref_blocks.append(ref_block)
                    ref_blocks_dst.append((dy, dx))
        return np.asarray(ref_blocks), ref_blocks_dst
    
    target_block_flat = target_block.flatten()
    ref_blocks, ref_blocks_dst = match(ref_frame, y, x, block_size, search_range)
    squared_difference = lambda x, y: (x - y) ** 2
    mses = np.mean(squared_difference(ref_blocks, target_block_flat), axis=1)
    min_idx = np.argmin(mses)

    return ref_blocks_dst[min_idx]

def block_matching_wrapper(args):
    return block_matching(*args)

def motion_estimation(ref_frame0, ref_frame1, target_blocks, block_size=16, search_range=7):
    with Pool() as pool:
        motion_vectors_ref0_to_target = pool.map(block_matching_wrapper, [(ref_frame0, tb, pos[0], pos[1], block_size, search_range) for tb, pos in target_blocks])
        motion_vectors_ref1_to_target = pool.map(block_matching_wrapper, [(ref_frame1, tb, pos[0], pos[1], block_size, search_range) for tb, pos in target_blocks])
    return np.array(motion_vectors_ref0_to_target), np.array(motion_vectors_ref1_to_target)

def global_motion_compensation(ref_frame0, ref_frame1, motion_vectors, block_size=16):
    height, width = ref_frame0.shape
    predicted_frame = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    num_of_blocks_x = width // block_size
    
    for i, ((dy0, dx0), (dy1, dx1)) in enumerate(zip(motion_vectors[0], motion_vectors[1])):
        y = (i // num_of_blocks_x) * block_size
        x = (i % num_of_blocks_x) * block_size

        y0, x0 = y + dy0, x + dx0
        y1, x1 = y + dy1, x + dx1

        if 0 <= y0 < height - block_size and 0 <= x0 < width - block_size:
            ref_block0 = ref_frame0[y0:y0 + block_size, x0:x0 + block_size]
            weights[y:y + block_size, x:x + block_size] += 1
            predicted_frame[y:y + block_size, x:x + block_size] += ref_block0 * (1 - (y0 % 1)) * (1 - (x0 % 1))
        if 0 <= y1 < height - block_size and 0 <= x1 < width - block_size:
            ref_block1 = ref_frame1[y1:y1 + block_size, x1:x1 + block_size]
            weights[y:y + block_size, x:x + block_size] += 1
            predicted_frame[y:y + block_size, x:x + block_size] += ref_block1 * (1 - (y1 % 1)) * (1 - (x1 % 1))

    non_zero_weights = weights > 0
    predicted_frame[non_zero_weights] /= weights[non_zero_weights]
    predicted_frame = np.clip(predicted_frame, 0, 255).astype(np.uint8)

    return predicted_frame