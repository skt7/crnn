# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:48:13 2018

@author: SKT
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 18:02:10 2018

@author: SKT
"""

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, Activation, Reshape, Dense, Lambda
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import SGD
import keras.losses

import cv2
import numpy as np
import itertools
import string
import matplotlib.pyplot as plt


ctc_loss = {'ctc': lambda y_true, y_pred: y_pred}
keras.losses.ctc_loss = ctc_loss

class CRNN:
    
    INPUT_SHAPE = (128, 32, 1)
    CHARS = string.ascii_lowercase + "\n"
    OUT_LEN = len(CHARS) + 1
    TEXT_LEN = 5 + 1
    N = 16
    model = None
    pred_model = None
        
    def _CNN(self, inputs):
        x = Conv2D(64, 3, padding='same', strides=1)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
    
        x = Conv2D(128, 3, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        
        x = Conv2D(256, 3, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        
        x = Conv2D(256, 3, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
    
        x = Conv2D(512, 3, padding='same', strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
       
        return x
    
    def _Map2Seq(self, inputs):    
        x = Reshape(target_shape=(self.N, -1))(inputs)
        return x
        
    def _RNN(self, inputs):
        
        n = 512
        
        gru_1 = GRU(n, return_sequences=True, name='gru1')(inputs)
        gru_1b = GRU(n, return_sequences=True, go_backwards=True, name='gru1_b')(inputs)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(n, return_sequences=True, name='gru2')(gru1_merged)
        gru_2b = GRU(n, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)  
        x = concatenate([gru_2, gru_2b]) 
        x = Dense(self.OUT_LEN, name='dense19')(x)
        x = Activation('softmax', name='activation20')(x)
        return x
        
    def ctc_lambda_func(self, args):
        pred, labels, input_length, label_length = args
        pred = pred[:, 2:, :]
        return K.ctc_batch_cost(labels, pred, input_length, label_length)
    
    def _CRNN(self):
        inputs = Input(shape=self.INPUT_SHAPE, name='inputs') 
        cnn = self._CNN(inputs)
        map2seq = self._Map2Seq(cnn)
        rnn = self._RNN(map2seq)
        
        #CTC
        labels = Input(name='labels', shape=[self.TEXT_LEN], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([rnn, labels, input_length, label_length])
        
        model = Model([inputs, labels, input_length, label_length], outputs=loss_out)
        
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        
        model.compile(loss=ctc_loss, optimizer=sgd)
#        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='sgd')
        
        return model
            
    def text_to_label(self, text):
        return np.array(list(map(lambda x: self.CHARS.index(x), text)))
    
    def process_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        img /= 255
        img = np.reshape(img, (self.INPUT_SHAPE))
        return img
    
                                
    def process_data(self, imgs, labels):
        inputs = np.array([self.process_img(img) for img in imgs])             
        labels = np.array([self.text_to_label(label + "\n") for label in labels])
        label_length = np.array([self.TEXT_LEN] * len(inputs))    
        input_length = np.ones((len(inputs), 1)) * (self.N-2)
        outputs = np.zeros([len(inputs)])
        return inputs, labels, input_length, label_length, outputs
    
    def train(self, imgs, labels, batch_size = 256, epochs = 20):
        inputs, labels, input_length, label_length, outputs = self.process_data(imgs, labels)
        self.model = self._CRNN()
        self.model.summary()
        self.model.fit([inputs, labels, input_length, label_length], 
              outputs,
              validation_split = 0.1,
              batch_size = batch_size, 
              epochs = epochs, 
              verbose = 1)
        
    def get_pred_model(self):
         net_inp = self.model.get_layer(name='inputs').input
         net_out = self.model.get_layer(name='activation20').output
         return Model(net_inp, net_out)
    
    def predict(self, img):
        
        if not self.pred_model:
            self.pred_model = self.get_pred_model()
        
        plt.imshow(img)
        plt.show()
        
        img = self.process_img(img)
        out = self.pred_model.predict(np.array([img]))    
        out_best = list(np.argmax(out[0, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(self.CHARS):
                outstr += self.CHARS[c]
        return outstr
    
    def save(self, name):
        self.model.save_weights(name + '.h5')
        
    def load(self, name):
        self.model = self._CRNN()
        self.model.load_weights(name + '.h5')
    
    
    
    
    
    
    
