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

import random
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

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam')

    model.summary()

    return model


def get_matrix(fname):
    img = cv2.imread(fname)
    return img 

# Generate data for training
def generate_data(data_names, labels ,bsize, dlen, reindex):
    splitpoint = int(0.8*dlen)
    i = 0
    while True:
        x = []
        y = []
        for j in range(i,i+bsize):
            ix = reindex[j]
            img = get_matrix(images[ix])
            lbl = np.array([labels[ix]])
            x.append(img)
            y.append(lbl)
        x = np.array(x)
        y = np.array(y)
        yield (x,y)
        i +=bsize
        if i+bsize > splitpoint:
            i = 0

# Generate data for validation
def generate_data_val(data_names, labels ,bsize, dlen, reindex):
    splitpoint = int(0.8*dlen)
    i = splitpoint
    while True:
        x = []
        y = []
        for j in range(i,i+bsize):
            ix = reindex[j]
            x.append(get_matrix(images[ix]))
            y.append(np.array([labels[ix]]))
        x = np.array(x)
        y = np.array(y)
        yield (x,y)
        i +=bsize
        if i+bsize > dlen:
            i = splitpoint


def train_model(data_names, labels, model, bsize=16, epochs=10):
    
    dlen = len(labels)
    splitpoint = int(0.8*dlen)
    reindex = list(range(len(labels)))
    random.seed(1234)
    random.shuffle(reindex)

    
    hs = model.fit_generator(generate_data(data_names, labels ,bsize, dlen, reindex),
                             steps_per_epoch=int(splitpoint/ bsize),
                             validation_data=generate_data_val(data_names, labels ,bsize, dlen, reindex),
                             validation_steps=(dlen-splitpoint)/bsize, epochs=epochs,callbacks=[model_checkpoint])
    