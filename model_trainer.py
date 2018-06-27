# Import libraries
import numpy as np

import matplotlib.pyplot as plt

import cv2

import pandas as pd

import tensorflow as tf

import keras
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

# (264, 198, 3)

def create_model(input_shape, num_of_classes=196):
    
    model = Sequential()

    # Feature Recognation
    model.add(Lambda(lambda x: (2*x / 255.0) - 1.0, input_shape=input_shape))

    model.add(Conv2D(16, (5,5), activation="relu", strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

    model.add(Conv2D(32, (5,5), activation="relu", strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

    model.add(Conv2D(64, (3,3), activation="relu", strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

    # Classification
    model.add(Flatten())

    model.add(Dense(512), activation='relu')

    model.add(Dense(216), activation='relu')

    model.add(Dense(216), activation='relu')

    model.add(Dense(num_of_classes, activation="softmax"))    

    return model
