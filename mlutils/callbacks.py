# -*- coding: utf-8 -*-
"""
Custom callbacks for Keras.
Created on Sat Jun  2 19:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/mlutils

"""


# imports
import os
import requests

from keras import callbacks
from matplotlib import pyplot
from random import randint


# Telegram class
class Telegram(callbacks.Callback):
    
    def __init__(self, auth_token, chat_id, monitor='val_acc', out_dir='.'):
        self._task_id = None
        self._chat_id = chat_id
        self._address = 'https://api.telegram.org/bot{}'.format(auth_token)
        self._history = {}
        self._monitor = {monitor.lower(): None}
        self._out_dir = out_dir
        return
    
    def on_train_begin(self, logs):
        self._task_id = randint(1000, 9999)
        for metric in self.params.get('metrics'):
            self._history[metric] = []
        data = {}
        data['chat_id'] = self._chat_id
        data['text'] = '`A new training has been started.\n[TASK ID: {}]`'.format(self._task_id)
        data['parse_mode'] = 'Markdown'
        self._send_message(data)
        return
    
    def on_train_end(self, logs):
        data = {}
        data['chat_id'] = self._chat_id
        data['text'] = '`An ongoing training is finished.\n[TASK ID: {}]`'.format(self._task_id)
        data['parse_mode'] = 'Markdown'
        self._send_message(data)
        self._plot()
        return
    
    def on_epoch_begin(self, epoch, logs):
        metric = next(iter(self._monitor))
        if self._monitor.get(metric) is None:
            data = {}
            data['chat_id'] = self._chat_id
            data['text'] = '`I am monitoring {} of an ongoing training. I will post an update whenever this metric improves.\n[TASK ID: {}]`'.format(metric, self._task_id)
            data['parse_mode'] = 'Markdown'
            self._send_message(data)
        return
    
    def on_epoch_end(self, epoch, logs):
        for metric in logs.keys():
            self._history[metric].append(logs.get(metric))
        metric = next(iter(self._monitor))
        old_value = self._monitor.get(metric)
        new_value = logs.get(metric)
        if old_value is None:
            self._monitor[metric] = new_value
            self._checkpoint(epoch, metric, old_value, new_value)
        elif metric.find('loss') >= 0 and new_value < old_value:
            self._monitor[metric] = new_value
            self._checkpoint(epoch, metric, old_value, new_value)
        elif metric.find('acc') >= 0 and new_value > old_value:
            self._monitor[metric] = new_value
            self._checkpoint(epoch, metric, old_value, new_value)
        return
    
    def on_batch_begin(self, batch, logs):
        return
    
    def on_batch_end(self, batch, logs):
        return
    
    def _checkpoint(self, epoch, metric, old_value, new_value):
        if metric.find('loss') >= 0 and old_value is None:
            old_value = '\u221e'
        elif metric.find('acc') >= 0 and old_value is None:
            old_value = '-\u221e'
        data = {}
        data['chat_id'] = self._chat_id
        if type(old_value) is str:
            data['text'] = '`{} of an ongoing training is improved from {} to {:.4f} after epoch {}.\n[TASK ID: {}]`'.format(metric, old_value, new_value, epoch+1, self._task_id)
        else:
            data['text'] = '`{} of an ongoing training is improved from {:.4f} to {:.4f} after epoch {}.\n[TASK ID: {}]`'.format(metric, old_value, new_value, epoch+1, self._task_id)
        data['parse_mode'] = 'Markdown'
        self._send_message(data)
        return
    
    def _plot(self):
        if not os.path.isdir(self._out_dir):
            os.makedirs(self._out_dir)
        plot_l = os.path.join(self._out_dir, 'plot_loss.png')
        plot_a = os.path.join(self._out_dir, 'plot_acc.png')
        
        # plot training and validation loss
        pyplot.figure()
        pyplot.title('Training and Validation Loss')
        pyplot.xlabel('epoch')
        pyplot.ylabel('loss')
        for metric in self._history.keys():
            if metric.find('loss') < 0:
                continue
            pyplot.plot(self._history[metric], label=metric)
        pyplot.legend()
        pyplot.savefig(plot_l)
        
        # plot training and validation accuracy
        pyplot.figure()
        pyplot.title('Training and Validation Accuracy')
        pyplot.xlabel('epoch')
        pyplot.ylabel('accuracy')
        for metric in self._history.keys():
            if metric.find('acc') < 0:
                continue
            pyplot.plot(self._history[metric], label=metric)
        pyplot.legend()
        pyplot.savefig(plot_a)
        
        # broadcast results
        data = {}
        data['chat_id'] = self._chat_id
        data['caption'] = '`Training and Validation Loss [TASK ID: {}]`'.format(self._task_id)
        data['parse_mode'] = 'Markdown'
        files = {}
        files['photo'] = open(plot_l, 'rb')
        self._send_photo(data, files)
        data['caption'] = '`Training and Validation Accuracy [TASK ID: {}]`'.format(self._task_id)
        files['photo'] = open(plot_a, 'rb')
        self._send_photo(data, files)
        return
    
    def _send_message(self, data):
        requests.post(self._address+'/sendMessage', data=data)
        return
    
    def _send_photo(self, data, files):
        requests.post(self._address+'/sendPhoto', data=data, files=files)
        return
