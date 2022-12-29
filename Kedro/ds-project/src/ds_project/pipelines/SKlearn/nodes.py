"""
This is a boilerplate pipeline 'SKlearn'
generated using Kedro 0.18.4
"""
import os
from os import listdir
from skimage.feature import hog
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random

#Creating features (flattened image and its HOG)
def create_features(img):
    color_features = np.array(img).flatten()
    # convert image to greyscale
    gray_image = ImageOps.grayscale(img)
    # get HOG features from greyscale image
    hog_features = hog(np.array(gray_image), block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack([color_features, hog_features])
    return flat_features

#We finalize creating the features for a sklearn model
def flatten_and_create_features(X):
    features_list = []
    for img in X:
        flat_features = create_features(img)
        features_list.append(flat_features)
    return np.array(features_list)

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
            
        X_train_res = [resize_image(img) for img in X_train]
        X_train_SK = flatten_and_create_features(X_train_res)
        y_train_SK = np.array(y_train)
        
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
        
        X_val, y_val = [], []
        
        for k in Validation_img:
            X_val.append(k[0])
            y_val.append(k[1])
            
        X_val_res = [resize_image(img) for img in X_val]
        X_val_SK = flatten_and_create_features(X_val_res)
        y_val_SK = np.array(y_val)
        
        return X_train_SK, y_train_SK, X_val_SK, y_val_SK


def build_and_train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    # Evaluate the model on the validation set
    accuracy = model.score(X_val, y_val)
    print("Accuracy:", accuracy)