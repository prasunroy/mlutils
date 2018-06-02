# -*- coding: utf-8 -*-
"""
Custom callbacks for Keras.
Created on Sat Jun  2 19:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/mlutils

"""


# imports
import requests

from keras import callbacks
from random import randint


# Telegram class
class Telegram(callbacks.Callback):
    def __init__(self, auth_token, chat_id, monitor='val_acc'):
        self._id = None
        self._url = 'https://api.telegram.org/bot{}'.format(auth_token)
        self._data = {'chat_id': chat_id}
        self._files = {}
        self._record = {}
        self._monitor = {monitor: None}
        
        return
    
    def on_train_begin(self, logs):
        for metric in self.params.get('metrics'):
            self._record[metric] = []
        self._id = randint(1000, 9999)
        self._data['text'] = '`A new training has been started [ID: {}].`'.format(self._id)
        self._data['parse_mode'] = 'Markdown'
        self._send_message()
        
        return
    
    def on_train_end(self, logs):
        self._data['text'] = '`An ongoing training is finished [ID: {}].`'.format(self._id)
        self._data['parse_mode'] = 'Markdown'
        self._id = None
        self._send_message()
        
        return
    
    def on_epoch_begin(self, epoch, logs):
        return
    
    def on_epoch_end(self, epoch, logs):
        return
    
    def on_batch_begin(self, batch, logs):
        return
    
    def on_batch_end(self, batch, logs):
        return
    
    def _send_message(self):
        requests.post(self._url+'/sendMessage', data=self._data)
        return
    
    def _send_photo(self):
        requests.post(self._url+'/sendPhoto', data=self._data,
                      files=self._files)
        return
