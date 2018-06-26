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