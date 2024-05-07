import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_image, vertical_projection,horizontal_projection


def lineSegmentation(preprocessed_image):
    horizontal_counts = horizontal_projection(preprocessed_image)
    zero_pixel_position = []

    zero_indices = np.where(horizontal_counts == 0)[0]

    # append continuous zero-pixel indices
    for idx in range(len(zero_indices)):
        if idx > 0 and idx < len(zero_indices) - 1:
            if zero_indices[idx - 1] == zero_indices[idx] - 1 or zero_indices[idx + 1] == zero_indices[idx] + 1:
                zero_pixel_position.append(zero_indices[idx])

    zero_pixel_position_temp = [] #stores the points for line segmentation

    for idx in range(len(zero_pixel_position) - 1):
        current_idx = zero_pixel_position[idx]
        next_idx = zero_pixel_position[idx + 1]
        if next_idx - current_idx == 1: 
            continue
        else:
            zero_pixel_position_temp.append(current_idx)
            zero_pixel_position_temp.append(next_idx)

    return zero_pixel_position_temp
