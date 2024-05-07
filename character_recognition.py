import cv2
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation import segment_characters,recognize_character,recognize_text,preprocess_image
import matplotlib.pyplot as plt

def characterRecognition(image_path,model,char_mapping):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    character_image= cv2.resize(image, (32, 32))
    

    normalized_char = np.array(character_image) / 255.0
    reshaped_char = np.expand_dims(normalized_char,axis=-1)
    reshaped_char=np.expand_dims(reshaped_char,axis=0)
    
    prediction = model.predict(reshaped_char)
    recognized_character_index = np.argmax(prediction)
    recognized_character = char_mapping[recognized_character_index]
    
    return recognized_character