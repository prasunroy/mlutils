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

from train_capsnet import build_model


# configurations
# -----------------------------------------------------------------------------
MODEL_LIST = ['capsnet', 'inceptionv3', 'mobilenet', 'resnet50',
              'vgg16', 'vgg19']
MODEL_DICT = {name.lower(): 'output/{}/checkpoints/{}_best.h5'\
              .format(name, name) for name in MODEL_LIST}
LABEL_MAPS = 'data/data_valid/labelmap.json'
IMAGE_DSRC = 'data/imgs_valid/'
IMAGE_READ = 1
OUTPUT_DIR = 'output/__test__/'
NOISE_LIST = ['Gaussian_White', 'Gaussian_Color', 'Salt_and_Pepper',
              'Gaussian_Blur', 'Motion_Blur', 'JPEG_Compression']
# -----------------------------------------------------------------------------

# setup parameters
sigmavals = [x for x in range(0, 256, 5)]
densities = [x/100 for x in range(0, 101, 5)]
gb_ksizes = [x for x in range(1, 52, 2)]
mb_ksizes = [x for x in range(3, 32, 2)]
qualities = [x for x in range(30, -1, -2)]
histories = {'y_'+name.lower(): [] for name in MODEL_LIST}


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
        if name.lower() == 'capsnet':
            models[name] = build_model()[1].load_weights(path)
        elif name.lower() == 'mobilenet':
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


# test models
def test():
    # validate paths
    if not validate_paths():
        return
    
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
            test_gaussian_white(x, y, models)
        elif noise.lower() == 'gaussian_color':
            test_gaussian_color(x, y, models)
        elif noise.lower() == 'salt_and_pepper':
            test_salt_and_pepper(x, y, models)
        elif noise.lower() == 'gaussian_blur':
            test_gaussian_blur(x, y, models)
        elif noise.lower() == 'motion_blur':
            test_motion_blur(x, y, models)
        elif noise.lower() == 'jpeg_compression':
            test_jpeg_compression(x, y, models)
    
    print('-'*35 + ' END TEST ' + '-'*35)
    
    return


# tests for gaussian white noise
def test_gaussian_white(x, y, models):
    # reset histories
    for name in histories.keys():
        histories[name] = []
    histories['x'] = sigmavals
    
    # run tests
    samples = len(x)
    for sigma in sigmavals:
        print('[INFO] Applying Gaussian white noise with mu=0 and sigma={}'\
              .format(sigma))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imnoise(image, 'gaussian_white', mu=0, sigma=sigma))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/samples), end='')
        test_models(noisy, y, models)
    
    # save and plot histories
    df = pandas.DataFrame(histories)
    df.to_csv(os.path.join(OUTPUT_DIR, 'gaussian_white.csv'), index=False)
    plot(filepath=os.path.join(OUTPUT_DIR, 'gaussian_white.png'),
         title='Change in test accuracy with Gaussian white noise',
         xlabel='standard deviation',
         ylabel='accuracy')
    
    return


# tests for gaussian color noise
def test_gaussian_color(x, y, models):
    # reset histories
    for name in histories.keys():
        histories[name] = []
    histories['x'] = sigmavals
    
    # run tests
    samples = len(x)
    for sigma in sigmavals:
        print('[INFO] Applying Gaussian color noise with mu=0 and sigma={}'\
              .format(sigma))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imnoise(image, 'gaussian_color', mu=0, sigma=sigma))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/samples), end='')
        test_models(noisy, y, models)
    
    # save and plot histories
    df = pandas.DataFrame(histories)
    df.to_csv(os.path.join(OUTPUT_DIR, 'gaussian_color.csv'), index=False)
    plot(filepath=os.path.join(OUTPUT_DIR, 'gaussian_color.png'),
         title='Change in test accuracy with Gaussian color noise',
         xlabel='standard deviation',
         ylabel='accuracy')
    
    return


# tests for salt and pepper noise
def test_salt_and_pepper(x, y, models):
    # reset histories
    for name in histories.keys():
        histories[name] = []
    histories['x'] = densities
    
    # run tests
    samples = len(x)
    for density in densities:
        print('[INFO] Applying salt and pepper noise with density={}'\
              .format(density))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imnoise(image, 'salt_and_pepper', density=density))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/samples), end='')
        test_models(noisy, y, models)
    
    # save and plot histories
    df = pandas.DataFrame(histories)
    df.to_csv(os.path.join(OUTPUT_DIR, 'salt_and_pepper.csv'), index=False)
    plot(filepath=os.path.join(OUTPUT_DIR, 'salt_and_pepper.png'),
         title='Change in test accuracy with salt and pepper noise',
         xlabel='noise density',
         ylabel='accuracy')
    
    return


# tests for gaussian blur
def test_gaussian_blur(x, y, models):
    # reset histories
    for name in histories.keys():
        histories[name] = []
    histories['x'] = gb_ksizes
    
    # run tests
    samples = len(x)
    for ksize in gb_ksizes:
        print('[INFO] Applying Gaussian blur with kernel size=({}, {})'\
              .format(ksize, ksize))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imnoise(image, 'gaussian_blur',
                                 gb_ksize=(ksize, ksize)))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/samples), end='')
        test_models(noisy, y, models)
    
    # save and plot histories
    df = pandas.DataFrame(histories)
    df.to_csv(os.path.join(OUTPUT_DIR, 'gaussian_blur.csv'), index=False)
    plot(filepath=os.path.join(OUTPUT_DIR, 'gaussian_blur.png'),
         title='Change in test accuracy with Gaussian blur',
         xlabel='kernel size',
         ylabel='accuracy')
    
    return


# tests for motion blur
def test_motion_blur(x, y, models):
    # reset histories
    for name in histories.keys():
        histories[name] = []
    histories['x'] = mb_ksizes
    
    # run tests
    samples = len(x)
    for ksize in mb_ksizes:
        print('[INFO] Applying motion blur with kernel size=({}, {})'\
              .format(ksize, ksize))
        noisy = []
        count = 0
        mb_kernel = numpy.zeros((ksize, ksize))
        mb_kernel[ksize//2, :] = 1
        mb_kernel /= numpy.sum(mb_kernel)
        for image in x:
            noisy.append(imnoise(image, 'motion_blur', mb_kernel=mb_kernel))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/samples), end='')
        test_models(noisy, y, models)
    
    # save and plot histories
    df = pandas.DataFrame(histories)
    df.to_csv(os.path.join(OUTPUT_DIR, 'motion_blur.csv'), index=False)
    plot(filepath=os.path.join(OUTPUT_DIR, 'motion_blur.png'),
         title='Change in test accuracy with motion blur',
         xlabel='kernel size',
         ylabel='accuracy')
    
    return


# tests for JPEG compression
def test_jpeg_compression(x, y, models):
    # reset histories
    for name in histories.keys():
        histories[name] = []
    histories['x'] = qualities
    
    # run tests
    samples = len(x)
    for quality in qualities:
        print('[INFO] Applying JPEG compression with quality={}'\
              .format(quality))
        noisy = []
        count = 0
        for image in x:
            noisy.append(imnoise(image, 'jpeg_compression', quality=quality))
            count += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(count*100/samples), end='')
        test_models(noisy, y, models)
    
    # save and plot histories
    df = pandas.DataFrame(histories)
    df.to_csv(os.path.join(OUTPUT_DIR, 'jpeg_compression.csv'), index=False)
    plot(filepath=os.path.join(OUTPUT_DIR, 'jpeg_compression.png'),
         title='Change in test accuracy with JPEG compression',
         xlabel='image quality',
         ylabel='accuracy', invert_xaxis=True)
    
    return


# test models
def test_models(x, y, models):
    print('')
    samples = len(x)
    for name in models.keys():
        print('[INFO] Preparing images for {}'.format(name))
        images = []
        counts = 0
        for image in x:
            images.append(cv2.resize(image, models[name].input_shape[1:3]))
            counts += 1
            print('\r[INFO] Progress... {:3.0f}%'\
                  .format(counts*100/samples), end='')
        print('\n[INFO] Testing images on {}... '.format(name), end='')
        x_test = numpy.asarray(images, dtype='float32') / 255.0
        y_test = numpy.asarray(y, dtype='int')
        if name == 'capsnet':
            p_test = models[name].predict(x_test)[0].argmax(axis=1)
        else:
            p_test = models[name].predict(x_test).argmax(axis=1)
        accuracy = sum(p_test==y_test)*100/samples
        histories['y_'+name].append(accuracy)
        print('done [accuracy: {:6.2f}%]'.format(accuracy))
    
    return


# plot histories
def plot(filepath, title='', xlabel='', ylabel='',
         invert_xaxis=False, invert_yaxis=False):
    pyplot.figure()
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    if invert_xaxis:
        pyplot.gca().invert_xaxis()
    if invert_yaxis:
        pyplot.gca().invert_yaxis()
    for name in histories.keys():
        if name == 'x':
            continue
        pyplot.plot(histories['x'], histories[name], label=name.split('_')[-1])
    pyplot.legend()
    pyplot.savefig(filepath)
    pyplot.show()
    
    return


# main
if __name__ == '__main__':
    test()
