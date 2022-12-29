"""
This is a boilerplate pipeline 'Keras'
generated using Kedro 0.18.4
"""
import os
from os import listdir
import numpy as np
import random
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras


#We resize the images to a 256x256 format
def resize_image(img):
    res = img.resize((256,256),Image.BICUBIC)
    return res



def load_data(train_dir, val_dir):
    Training_img = []
        
    Dmg_train_dir = train_dir + "/00-damage"
        
    for images in os.listdir(Dmg_train_dir):
        im = Image.open(Dmg_train_dir + '/' + images)
        Training_img.append([im,1])
            
    Whole_train_dir = train_dir + "/01-whole"
        
    for images in os.listdir(Whole_train_dir):
        im = Image.open(Whole_train_dir + '/' + images)
        Training_img.append([im,0])
            
    random.shuffle(Training_img)
        
    #We needed to take images and labels as a whole as to shuffle them so as to avoid any bias when training models
        
    X_train,y_train = [],[]
        
    for k in Training_img:
        X_train.append(k[0])
        y_train.append(k[1])
        
    X_train_keras = np.array([np.asarray(resize_image(img)) for img in X_train])
    y_train_keras = np.array(y_train)
        
    Validation_img = []
        
    Dmg_val_dir = val_dir + "/00-damage"
        
    for images in os.listdir(Dmg_val_dir):
        im = Image.open(Dmg_val_dir + '/' + images)
        Validation_img.append([im,1])
            
    Whole_val_dir = val_dir + "/01-whole"
        
    for images in os.listdir(Whole_val_dir):
        im = Image.open(Whole_val_dir + '/' + images)
        Validation_img.append([im,0])
            
    random.shuffle(Validation_img)
        
    #Same reason as above for this part of the code
        
    for k in Validation_img:
        X_val.append(k[0])
        y_val.append(k[1])
            
    X_val_keras = np.array([np.asarray(resize_image(img)) for img in X_val])
    y_val_keras = np.array(y_val)
        
    return X_train_keras, y_train_keras, X_val_keras, y_val_keras

# model building and training function

def build_and_train_model(X_train, y_train):
    # Build the model
    # We chose a LeNet-5 architecture of CNN for two reasons:
    # First, we think it is a nice reference to the work of a great french researcher, Mr Yann LeCun. 
    # Second, since it was one the first CNN model it is surely not the most efficient to date and thus we can have comparison with more classical ML models 
    # (less designated for object detection than CNNs) 
    
    model = keras.Sequential([
        keras.layers.Conv2D(6, (5, 5), padding='same', input_shape=(256, 256, 3)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(16, (5, 5)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(120),
        keras.layers.Activation('relu'),
        keras.layers.Dense(84),
        keras.layers.Activation('relu'),
        keras.layers.Dense(1),
        keras.layers.Activation('sigmoid')
    ])

    # Compile the model with loss and optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    
    return model

def evaluate_model(model, X_val, y_val):
    # Evaluate the model on the validation set
    evaluation = model.evaluate(X_val, y_val, batch_size=32)
    return evaluation

                  