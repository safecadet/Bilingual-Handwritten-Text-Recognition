import cv2
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import preprocess_image, vertical_projection,horizontal_projection, identify_hindi_word
from character_segmentation import characterSegmentation
from line_segmentation import lineSegmentation
from word_segmentation import wordSegmentation
from character_recognition import characterRecognition

model = load_model("model.h5")

char_mapping = [
    'O', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'अ',
    'अं', 'अः', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'कं', 'कः', 'का', 'कि',
    'की', 'कु', 'कू', 'के', 'कै', 'को', 'कौ', 'ख', 'खं', 'खः', 'खा', 'खि', 'खी', 'खु', 'खू',
    'खे', 'खै', 'खो', 'खौ', 'ग', 'गं', 'गः', 'गा', 'गि', 'गी', 'गु', 'गू', 'गे', 'गै', 'गो',
    'गौ', 'घ', 'घं', 'घः', 'घा', 'घि', 'घी', 'घु', 'घू', 'घे', 'घै', 'घो', 'ङ', 'च', 'चं',
    'चः', 'चा', 'चि', 'ची', 'चु', 'चू', 'चे', 'चै', 'चो', 'चौ', 'छ', 'छं', 'छः', 'छा', 'छि',
    'छी', 'छु', 'छू', 'छे', 'छै', 'छो', 'छौ', 'ज', 'जं', 'जः', 'जा', 'जि', 'जी', 'जु', 'जू',
    'जे', 'जै', 'जो', 'जौ', 'झ', 'झं', 'झः', 'झा', 'झि', 'झी', 'झु', 'झू', 'झे', 'झै', 'झो',
    'झौ', 'ञ', 'ट', 'टं', 'टः', 'टा', 'टि', 'टी', 'टु', 'टू', 'टे', 'टै', 'टो', 'टौ', 'ठ',
    'ठं', 'ठः', 'ठा', 'ठि', 'ठी', 'ठु', 'ठू', 'ठे', 'ठै', 'ठो', 'ठौ', 'ड', 'डं', 'डः', 'डा',
    'डि', 'डी', 'डु', 'डू', 'डे', 'डै', 'डो', 'डौ', 'ढ', 'ढं', 'ढः', 'ढा', 'ढि', 'ढी', 'ढु',
    'ढू', 'ढे', 'ढै', 'ढो', 'ढौ', 'ण', 'णं', 'णः', 'णा', 'णि', 'णी', 'णु', 'णू', 'णे', 'णै',
    'णो', 'णौ', 'त', 'तं', 'तः', 'ता', 'ति', 'ती', 'तु', 'तू', 'ते', 'तै', 'तो', 'तौ', 'थ',
    'थं', 'थः', 'था', 'थि', 'थी', 'थु', 'थू', 'थे', 'थै', 'थो', 'थौ', 'द', 'दं', 'दः', 'दा',
    'दि', 'दी', 'दु', 'दू', 'दे', 'दै', 'दो', 'दौ', 'ध', 'धं', 'धः', 'धा', 'धि', 'धी', 'धु',
    'धू', 'धे', 'धै', 'धो', 'धौ', 'न', 'नं', 'नः', 'ना', 'नि', 'नी', 'नु', 'नू', 'ने', 'नै',
    'नो', 'नौ', 'प', 'पं', 'पः', 'पा', 'पि', 'पी', 'पु', 'पू', 'पे', 'पै', 'पो', 'पौ', 'फ',
    'फं', 'फः', 'फा', 'फि', 'फी', 'फु', 'फू', 'फे', 'फै', 'फो', 'फौ', 'ब', 'बं', 'बः', 'बा',
    'बि', 'बी', 'बु', 'बू', 'बे', 'बै', 'बो', 'बौ', 'भ', 'भं', 'भः', 'भा', 'भि', 'भी', 'भु',
    'भू', 'भे', 'भै', 'भो', 'भौ', 'म', 'मं', 'मः', 'मा', 'मि', 'मी', 'मु', 'मू', 'मे', 'मै',
    'मो', 'मौ', 'य', 'यं', 'यः', 'या', 'यि', 'यी', 'यु', 'यू', 'ये', 'यै', 'यो', 'यौ', 'र',
    'रं', 'रः', 'रा', 'रि', 'री', 'रे', 'रै', 'रो', 'रौ', 'ल', 'लं', 'लः', 'ला', 'लि', 'ली',
    'लु', 'लू', 'ले', 'लै', 'लो', 'लौ', 'व', 'वं', 'वः', 'वा', 'वि', 'वी', 'वु', 'वू', 'वे',
    'वै', 'वो', 'वौ', 'श', 'शं', 'शः', 'शा', 'शि', 'शी', 'शु', 'शू', 'शे', 'शै', 'शो', 'शौ',
    'ष', 'षं', 'षः', 'षा', 'षि', 'षी', 'षु', 'षू', 'षे', 'षै', 'षो', 'षौ', 'स', 'सं', 'सः',
    'सा', 'सि', 'सी', 'सु', 'सू', 'से', 'सै', 'सो', 'सौ', 'ह', 'हं', 'हः', 'हा', 'हि', 'ही',
    'हु', 'हू', 'हे', 'है', 'हो', 'हौ'
]

image_path ='test_images/test_image_8.jpg'
input_image, preprocessed_image = preprocess_image(image_path)
save_folder = 'segmented images'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    file_list = os.listdir(save_folder)
    for file_name in file_list:
        file_path = os.path.join(save_folder, file_name)
        os.remove(file_path)
        
line_seg_array = lineSegmentation(preprocessed_image) #store row numbers for segmenting lines

for idx in range(0, len(line_seg_array) - 1, 2):
    line_cropped_image = preprocessed_image[line_seg_array[idx]:line_seg_array[idx + 1] , :]
    word_seg_array = wordSegmentation(line_cropped_image) #store column numbers for segmenting words
    
    for idx1 in range(0, len(word_seg_array) - 1, 1):
        word_cropped_image = line_cropped_image[:, word_seg_array[idx1]:word_seg_array[idx1+1]]
        character_seg_array = characterSegmentation(word_cropped_image) #store column numbers for segmenting characters
            
        for idx2 in range(0, len(character_seg_array) - 1, 2):
            if abs(character_seg_array[idx2] - character_seg_array[idx2 + 1]) > 5: #if the detected character is more than 5 pixels, then most probably it's not a noise due to segmentation
                character_cropped_image = word_cropped_image[:, character_seg_array[idx2]:character_seg_array[idx2 + 1]]
            else:
                continue
            
            #crop the segmented character image from top and bottom to remove any black bars
            horizontal_counts = horizontal_projection(character_cropped_image)
            top_index = np.min(np.where(horizontal_counts != 0))
            bottom_index = np.max(np.where(horizontal_counts != 0))
            character_cropped_image = character_cropped_image[top_index:bottom_index, :]
            
            save_path = os.path.join(save_folder, f'line{idx//2}_word{idx1}_character{idx2//2}.png')
            if character_cropped_image.shape != (32, 32):
                pad_width = max(0, 32 - character_cropped_image.shape[0])
                pad_height = max(0, 32 - character_cropped_image.shape[1])
                
                padded_image = np.pad(character_cropped_image, ((8, 8), (8, 8)), mode='constant')
                resized_image = cv2.resize(padded_image, (32, 32))
                
                cv2.imwrite(save_path, resized_image)
            else:
                cv2.imwrite(save_path, character_cropped_image)


with open("output.txt", "w", encoding="utf-8") as file:
    prev_line = 0
    prev_word = 0
    for image_file in os.listdir(save_folder):
        image_path = os.path.join(save_folder, image_file)
        text = characterRecognition(image_path, model, char_mapping)
        line_idx = int(image_file.split('_')[0][4:])
        word_idx = int(image_file.split('_')[1][4:])
        character_idx = int(image_file.split('_')[2][9:-4])
        
        if line_idx != prev_line:
            file.write('\n')
            prev_line = line_idx
        elif prev_word != word_idx:
            prev_word = word_idx
            file.write(' ')
            
        
        file.write(text)