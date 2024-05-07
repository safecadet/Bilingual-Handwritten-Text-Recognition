import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_thresholding(gray):

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def remove_noise(image):
    thresh = otsu_thresholding(image)
    
    result = cv2.bitwise_and(image, image, mask=thresh)
    
    denoised_image = cv2.medianBlur(result, 3)
    
    return denoised_image


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image = remove_noise(binary_image) 

    return image, binary_image

def horizontal_projection(binary_image):
    height = binary_image.shape[0]

    horizontal_counts = np.zeros(height)

    for row in range(height):
        horizontal_counts[row] = np.sum(binary_image[row, :] == 255)

    return horizontal_counts


def vertical_projection(binary_image):
    width = binary_image.shape[1]

    vertical_counts = np.zeros(width)

    for col in range(width):
        vertical_counts[col] = np.sum(binary_image[:, col] == 255)

    return vertical_counts

def identify_hindi_word(binary_image):
    vertical_counts = vertical_projection(binary_image)

    if np.all(vertical_counts > 0):
        return True 
    else:
        return False