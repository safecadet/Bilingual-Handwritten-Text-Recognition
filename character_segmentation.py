import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_image, vertical_projection, identify_hindi_word,horizontal_projection


def characterSegmentation(preprocessed_image):
    vertical_counts = vertical_projection(preprocessed_image)

    
    zero_indices_pre = np.where(vertical_counts == 0)[0]
    first_non_zero_index = np.min(np.where(vertical_counts != 0))
    last_non_zero_index = np.max(np.where(vertical_counts != 0))
    
    preprocessed_image = preprocessed_image[:, first_non_zero_index:last_non_zero_index]
    
    zero_indices = []
    if identify_hindi_word(preprocessed_image):
        zero_indices = np.where(vertical_counts <= 4)[0]
        
    else:
        zero_indices = np.where(vertical_counts == 0)[0]
    
    zero_pixel_position = []
    
    for idx in range(len(zero_indices)):
        if idx > 0 and idx < len(zero_indices) - 1:
            if zero_indices[idx - 1] == zero_indices[idx] - 1 or zero_indices[idx + 1] == zero_indices[idx] + 1:
                zero_pixel_position.append(zero_indices[idx])
               
    zero_pixel_position_temp = []
    for idx in range(len(zero_pixel_position) - 1):
        current_idx = zero_pixel_position[idx]
        next_idx = zero_pixel_position[idx + 1]
        if next_idx - current_idx == 1:
            continue
        else:
            zero_pixel_position_temp.append(current_idx)
            zero_pixel_position_temp.append(next_idx)

    return zero_pixel_position_temp
