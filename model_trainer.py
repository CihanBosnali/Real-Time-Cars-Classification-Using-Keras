# Import libraries
import numpy as np

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD

import random

import glob
import json

def get_data(data_path="./data/cars_train/",
             labels_path="./data/devkit/train_perfect_preds.txt"):
    data = glob.glob(data_path+"*.jpg")
    with open(labels_path, "r") as labelsfile:
        labels = labelsfile.read().split("\n")
    return data, labels

# This function creates the architacture of our model
# Input shape? (264, 198, 3)
def create_model(input_shape, num_of_classes=196):
    
    model = Sequential()

    # Feature Recognation
    model.add(Lambda(lambda x: (2*x / 255.0) - 1.0, input_shape=input_shape))

    model.add(Conv2D(32, (5,5), activation="relu", strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same"))

    model.add(Conv2D(64, (5,5), activation="relu", strides=(2,2)))
    
    model.add(Conv2D(64, (3,3), activation="relu", strides=(2,2)))
    
    model.add(Conv2D(128, (2,2), activation="relu", strides=(2,2)))

    # Classification
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(216, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_of_classes, activation="softmax"))    

    sgd = SGD(lr=0.01, clipvalue=0.5)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd)

    model.summary()

    return model


def get_matrix(fname):
    img = cv2.imread(fname)
    img = cv2.resize(img, (200, 250))
    return img 

# Generate data for training
def generate_data(data_names, labels ,bsize, dlen, reindex):
    splitpoint = int(0.8*dlen)
    i = 0
    while True:
        x = []
        y = []
        for j in range(i,i+bsize):
            ix = reindex[j] - 1
            # -1 because dataset starts from 1 but arrays from 0
            img = get_matrix(data_names[ix])
            l = int(labels[ix]) - 1
            lbl = np.zeros((196))
            lbl[l] = 1
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
            ix = reindex[j] - 1
            # -1 because dataset starts from 1 but arrays from 0
            img = get_matrix(data_names[ix])
            l = int(labels[ix]) - 1
            lbl = np.zeros((196))
            lbl[l] = 1
            x.append(img)
            y.append(lbl)
        x = np.array(x)
        y = np.array(y)        
        yield (x,y)
        i +=bsize
        if i+bsize > dlen:
            i = splitpoint

# Generate data for validation
def generate_data_before(data_names, labels ,bsize, dlen, reindex):
    splitpoint = int(0.9*dlen)
    
    for i in range(splitpoint, dlen):
        x = []
        y = []
        ix = reindex[i] - 1
        # -1 because dataset starts from 1 but arrays from 0
        img = get_matrix(data_names[ix])
        l = int(labels[ix]) - 1
        lbl = np.zeros((196))
        lbl[l] = 1
        x.append(img)
        y.append(lbl)
    x = np.array(x)
    y = np.array(y)        
    return (x,y)

def plot_training(hs):
    # Train and validation loss chart
    print(hs.history.keys())
    
    plt.plot(hs.history['loss'])
    plt.plot(hs.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def train_model(data_names, labels, model, bsize=16, epochs=10):
    
    dlen = len(labels)
    splitpoint = int(0.8*dlen)
    reindex = list(range(len(labels)))
    random.seed(1234)
    random.shuffle(reindex)

    Board_Callback = TensorBoard(log_dir='./logs',
                                 histogram_freq=1,
                                 batch_size=16,
                                 write_graph=True,
                                 write_grads=True,
                                 write_images=True,
                                 embeddings_freq=0,
                                 embeddings_layer_names=None)

    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    valdata = generate_data_before(data_names, labels ,bsize, dlen, reindex)
    print(len(valdata))
    hs = model.fit_generator(generate_data(data_names, labels ,bsize, dlen, reindex),
                             steps_per_epoch=int(splitpoint/ bsize),
                             validation_data=valdata,
                             validation_steps=(dlen-splitpoint)/bsize, epochs=epochs,callbacks=[model_checkpoint, Board_Callback])

    plot_training(hs)

    return model
    
def save_model(model, mname="model_new"):
    # Save model weights and json.
    model.save_weights(mname+'.h5')
    model_json  = model.to_json()
    with open(mname+'.json', 'w') as outfile:
        json.dump(model_json, outfile)

def run_training():
    data, labels = get_data()
    model = create_model(input_shape=(250,200,3))
    model = train_model(data, labels, model, bsize=16, epochs=10)
    save_model(model, mname="car_model")

if __name__ == "__main__":
    run_training()