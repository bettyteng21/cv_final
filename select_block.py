import numpy as np

def select_blocks(blocks, reference_blocks, num_blocks):
    scores = np.zeros(len(blocks))

    for idx, (block,(y,x)) in enumerate(blocks):
        (ref_block,(a,b)) = reference_blocks[idx]
        mse = np.mean((block - ref_block) ** 2)
        scores[idx] = mse


    sorted_indices = np.argsort(scores)
    output_list = [0] * len(blocks)

    for i in sorted_indices[:num_blocks]:
        output_list[i] = 1

    return output_list