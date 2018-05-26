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

import cv2
import glob
import json
import numpy
import os
import pandas
import random

from keras.applications import mobilenet
from keras.models import load_model
from matplotlib import pyplot


# configurations
# -----------------------------------------------------------------------------
MODEL_LIST = ['inceptionv3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19']
MODEL_DICT = {name.lower(): 'output/{}/checkpoints/{}_best.h5'\
              .format(name, name) for name in MODEL_LIST}
LABEL_MAPS = 'data/data_valid/labelmap.json'
IMAGE_DSRC = 'data/imgs_valid/'
IMAGE_READ = 1
OUTPUT_DIR = 'output/__test__/'
NOISE_LIST = ['Gaussian_White', 'Gaussian_Color', 'Salt_and_Pepper',
              'Gaussian_Blur', 'Motion_Blur', 'JPEG_Compression']
# -----------------------------------------------------------------------------


# validate paths
def validate_paths():
    for name, path in MODEL_DICT.items():
        if not os.path.isfile(path):
            print('[INFO] Model checkpoint not found at {}'.format(path))
            return False
    if not os.path.isfile(LABEL_MAPS):
        print('[INFO] Label mapping not found')
        return False
    if not os.path.isdir(IMAGE_DSRC):
        print('[INFO] Image data source not found')
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
            image = cv2.imread(file, IMAGE_READ)
            if image is None:
                continue
            x.append(image)
            y.append(labelmap[label])
    
    return (x, y)


# load models
def load_models():
    models = {}
    for name, path in MODEL_DICT.items():
        if name.lower() == 'mobilenet':
            models[name] = load_model(path, custom_objects={'relu6': mobilenet.relu6})
        else:
            models[name] = load_model(path)
    
    return models


# apply noise
def imnoise(image, model, mu=0, sigma=0, density=0, gb_ksize=(1, 1),
            mb_kernel=numpy.zeros((1, 1), dtype='uint8'), quality=100):
    image = image.copy()
    if len(image.shape) == 2:
        image = numpy.expand_dims(image, 2)
    
    # get dimension of the image
    h, w, c = image.shape
    
    # apply a noise model
    model = model.lower()
    
    if model == 'gaussian_white':
        noise = numpy.random.normal(mu, sigma, (h, w))
        noise = numpy.dstack([noise]*c)
        image = image + noise
        image = cv2.normalize(image, None, 0, 255,
                              cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    elif model == 'gaussian_color':
        noise = numpy.random.normal(mu, sigma, (h, w, c))
        image = image + noise
        image = cv2.normalize(image, None, 0, 255,
                              cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    elif model == 'salt_and_pepper':
        if density < 0:
            density = 0
        elif density > 1:
            density = 1
        x = random.sample(range(w), w)
        y = random.sample(range(h), h)
        x, y = numpy.meshgrid(x, y)
        xy = numpy.c_[x.reshape(-1), y.reshape(-1)]
        n = int(w * h * density)
        n = random.sample(range(w*h), n)
        for i in n:
            if random.random() > 0.5:
                image[xy[i][1], xy[i][0], :] = 255
            else:
                image[xy[i][1], xy[i][0], :] = 0
    
    elif model == 'gaussian_blur':
        image = cv2.GaussianBlur(image, gb_ksize, 0)
    
    elif model == 'motion_blur':
        image = cv2.filter2D(image, -1, mb_kernel)
    
    elif model == 'jpeg_compression':
        if quality < 0:
            quality = 0
        elif quality > 100:
            quality = 100
        image = cv2.imencode('.jpg', image,
                             [int(cv2.IMWRITE_JPEG_QUALITY), quality])[-1]
        image = cv2.imdecode(image, -1)
    
    if image.shape[-1] == 1:
        image = numpy.squeeze(image, 2)
    
    return image


# test model
def test():
    # load data
    print('[INFO] Loading data... ', end='')
    (x, y) = load_data()
    print('done')
    
    # load models
    print('[INFO] Loading models... ', end='')
    models = load_models()
    print('done')
    
    print('-'*34 + ' BEGIN TEST ' + '-'*34)
    
    # run tests
    for noise in NOISE_LIST:
        if noise.lower() == 'gaussian_white':
            pass
        elif noise.lower() == 'gaussian_color':
            pass
        elif noise.lower() == 'salt_and_pepper':
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
