import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_image, vertical_projection,horizontal_projection,identify_hindi_word


def wordSegmentation(preprocessed_image):
    vertical_counts = vertical_projection(preprocessed_image)
    zero_pixel_position = []

    zero_indices = np.where(vertical_counts == 0)[0]

    for idx in range(len(zero_indices)):
        if idx > 0 and idx < len(zero_indices) - 1:
            if zero_indices[idx - 1] == zero_indices[idx] - 1 or zero_indices[idx + 1] == zero_indices[idx] + 1:
                zero_pixel_position.append(zero_indices[idx])
    
    zero_pixel_position_temp = []
    zero_pixel_position_temp.append(zero_pixel_position[0])
    for idx in range(len(zero_pixel_position) - 1):
        current_idx = zero_pixel_position[idx]
        next_idx = zero_pixel_position[idx + 1]
        if next_idx - current_idx == 1:
            continue
        else:
            zero_pixel_position_temp.append(current_idx)
            zero_pixel_position_temp.append(next_idx)
    zero_pixel_position_temp.append(zero_pixel_position[-1])
            
    seg_word_position = []
    gap_threshold = 25
    for idx in range(0, len(zero_pixel_position_temp) - 1, 2):
        current_idx = zero_pixel_position_temp[idx]
        next_idx = zero_pixel_position_temp[idx + 1]
        if next_idx - current_idx < gap_threshold:
            continue
        else:
            gap_point = int((current_idx + next_idx) // 2)
            seg_word_position.append(gap_point)

    return seg_word_position

