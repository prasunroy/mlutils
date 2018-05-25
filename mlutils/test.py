# -*- coding: utf-8 -*-
"""
Performance test of deep convolutional neural networks against various noises.
Created on Thu May 24 11:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/mlutils

"""


# imports
from __future__ import division
from __future__ import print_function

import glob
import json
import numpy
import os
import pandas

from keras.applications import mobilenet
from keras.models import load_model
from matplotlib import pyplot

from cvutils.io import imread
from cvutils.geometric import scale
from cvutils.noise import imnoise


# configurations
# -----------------------------------------------------------------------------
MODEL_NAME = 'vgg19'
MODEL_CKPT = 'output_{}/checkpoints/{}_best.h5'.format(MODEL_NAME, MODEL_NAME)
IMAGE_DSRC = 'data/imgs_valid/'
IMAGE_RFLG = 1
LABEL_MAPS = 'data/data_valid/labelmap.json'
OUTPUT_DIR = 'output_{}/test/'.format(MODEL_NAME)
# -----------------------------------------------------------------------------

NOISE_LIST = ['Gaussian_White',
              'Gaussian_Color',
              'Salt_and_Pepper',
              'Gaussian_Blur',
              'Motion_Blur',
              'JPEG_Compression']


# validate paths
def validate_paths():
    if not os.path.isfile(MODEL_CKPT):
        print('[INFO] Model checkpoint not found')
        return False
    if not os.path.isdir(IMAGE_DSRC):
        print('[INFO] Validation data not found')
        return False
    if not os.path.isfile(LABEL_MAPS):
        print('[INFO] Label mapping not found')
        return False
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    elif len(glob.glob(os.path.join(OUTPUT_DIR, '*.*'))) > 0:
        print('[INFO] Output directory must be empty')
        return False
    
    return True


# load data
def load_data():
    x = []
    y = []
    
    # label mapping
    with open(LABEL_MAPS, 'r') as file:
        labelmap = json.load(file)
    
    # class labels
    labels = [os.path.split(d[0])[-1] for d in os.walk(IMAGE_DSRC)][1:]
    
    # read images
    for label in labels:
        for file in glob.glob(os.path.join(IMAGE_DSRC, label, '*.*')):
            image = imread(file, IMAGE_RFLG)
            if image is None:
                continue
            x.append(image)
            y.append(labelmap[label])
    
    return (x, y)


# load model checkpoint
def load_ckpt():
    if MODEL_NAME.lower() == 'mobilenet':
        model = load_model(MODEL_CKPT, custom_objects={'relu6': mobilenet.relu6})
    else:
        model = load_model(MODEL_CKPT)
    
    return model


# test model
def test():
    # load data
    print('[INFO] Loading data... ', end='')
    (x, y) = load_data()
    print('done')
    
    # load model checkpoint
    print('[INFO] Loading model... ', end='')
    model = load_ckpt()
    print('done')
    model.summary()
    
    print('-'*34 + ' BEGIN TEST ' + '-'*34)
    
    # run tests
    for noise in NOISE_LIST:
        if noise.lower() == 'gaussian_white':
            pass
        elif noise.lower() == 'gaussian_color':
            pass
        elif noise.lower() == 'salt-and-pepper':
            pass
        elif noise.lower() == 'gaussian_blur':
            pass
        elif noise.lower() == 'motion_blur':
            pass
        elif noise.lower() == 'jpeg_compression':
            pass
    
    print('-'*35 + ' END TEST ' + '-'*35)
    
    return


# main
if __name__ == '__main__':
    test()
