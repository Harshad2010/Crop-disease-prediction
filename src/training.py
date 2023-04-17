# import libraries

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
from pathlib import Path
from src.exception import CropException
import sys

class TrainModel:
    def __init__():
        pass

    def train():
        '''It will train the model'''
        try:

            TRAIN_PATH = "./Crop-disease-prediction/src/dataset/train"
            VALIDATION_PATH = "./Crop-disease-prediction/src/dataset/val"          
                        
            #Creating the Pathlib PATH objects
            train_data_path = Path(TRAIN_PATH)
            validation_data_path = Path(VALIDATION_PATH)

            from tensorflow.python import train
            # this is the augmentation configuration we will use for training
            # It generate more images using below parameters.
            training_datagen = ImageDataGenerator(
                                                rescale = 1./255,
                                                rotation_range = 40,
                                                width_shift_range=0.2 ,  
                                                height_shift_range= 0.2, 
                                                shear_range = 0.2, 
                                                zoom_range = 0.2,
                                                horizontal_flip =True,
                                                fill_mode="nearest")

            valid_datagen = ImageDataGenerator(rescale=1./255)


            training_data = training_datagen.flow_from_directory( train_data_path, #this is target directory
                                                                target_size=(150,150), #all images will be resized to 150x150
                                                                batch_size=32,
                                                                class_mode="binary")

            valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                                        target_size = (150,150),
                                                        batch_size=32,
                                                        class_mode="binary")

            # Save the model using val accuracy
            model_path = "./Crop-disease-prediction/cotton_pant_disease_prediction_ai/model/v3_pred_cott_dis.h5"
            checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
            callbacks_list=[checkpoint]

            # Building CNN model

            #not able to image how pool size is implement after convolution of filters on image--just check
            cnn_model= keras.models.Sequential([
                                                keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150,150,3]),
                                                keras.layers.MaxPooling2D(pool_size=(2,2)),

                                                keras.layers.Conv2D(filters=64, kernel_size=3),
                                                keras.layers.MaxPooling2D(pool_size=(2,2)),

                                                keras.layers.Conv2D(filters=128, kernel_size=3),
                                                keras.layers.MaxPooling2D(pool_size=(2,2)),

                                                keras.layers.Conv2D(filters=256, kernel_size=3),
                                                keras.layers.MaxPooling2D(pool_size=(2,2)),

                                                keras.layers.Dropout(0.5),
                                                keras.layers.Flatten(), #neural network building
                                                
                                                keras.layers.Dense(units=128, activation='relu'), #input layers
                                                keras.layers.Dropout(0.1),

                                                keras.layers.Dense(units=256, activation='relu'),
                                                keras.layers.Dropout(0.25),

                                                keras.layers.Dense(units=4, activation='softmax') #output layer

                                            ])

            # Compile CNN model
            cnn_model.compile(loss="sparse_categorical_crossentropy", metrics=['accuracy'], optimizer= Adam(lr=0.0001))

            # train cnn model
            history = cnn_model.fit(training_data,
                                    epochs=10,
                                    verbose=1,
                                    validation_data=valid_data,
                                    callbacks=callbacks_list)

            model_path2 = "./Crop-disease-prediction/cotton_pant_disease_prediction_ai/model2/v4_1_pred_cott_dis.h5" # replace your path
            cnn_model.save(model_path2)
            
        except Exception as e:
                raise CropException(e,sys)

