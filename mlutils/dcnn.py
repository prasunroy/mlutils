# -*- coding: utf-8 -*-
"""
Deep convolutional neural networks.
Created on Tue May 22 20:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/mlutils

"""


# imports
from __future__ import division
from __future__ import print_function

import cv2
import numpy
import os

from keras import applications
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.utils import to_categorical
from scipy.io import loadmat
from scipy.io import savemat


# configurations
ARCHITECTURE = 'inceptionv3'
INCLUDE_TOP  = True
WEIGHTS_INIT = 'imagenet'
INPUT_TENSOR = None
INPUT_SHAPE  = None
NUM_CLASSES  = 1000
FREEZE_LAYER = 0
DATA_TRAIN = 'data_train/data.h5'
DATA_VALID = 'data_valid/data.h5'
IMAGE_SIZE = (100, 100, 3)
BATCH_SIZE = 100
NUM_EPOCHS = 100
OUTPUT_DIR = 'output_{}/'.format(ARCHITECTURE)
