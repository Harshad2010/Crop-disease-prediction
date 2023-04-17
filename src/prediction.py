import numpy as np
import os
import sys
from src.exception import CropException
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = load_model('model2/v4_1_pred_cott_dis.h5')

CLASS_NAMES = ['diseased cotton leaf','diseased cotton plant','fresh cotton leaf','fresh cotton plant']
 
class Prediction:
    def __init__(self):
        pass
 
    # Function to predict cotton disease
    def pred_cot_dieas(cott_plant):
        ''' after uploading x-ray image it will predict the result'''
        try:
            test_image = load_img(cott_plant, target_size = (150, 150)) # load image 
            test_image = img_to_array(test_image)/255 # convert image to np array and normalize
            test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
            result = model.predict(test_image).round(3) # predict diseased palnt or not
            
            pred = np.argmax(result) # get the index of max value
            
            if pred == 0:
                pred = CLASS_NAMES[0]
            elif pred == 1:
                pred = CLASS_NAMES[1]
            elif pred == 2:
                pred = CLASS_NAMES[2]
            else:
                pred = CLASS_NAMES[3]
        
        except Exception as e:
            raise CropException(e,sys)

#------------>>pred_cot_dieas<<--end