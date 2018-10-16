# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:50:36 2018

@author: SKT
"""

import cv2
import numpy as np
import string

class DataGen:    
    
    n_words = 0
    word_len = 0
    n_samples = 0
    
    def __init__(self, n_words = 1000, word_len=5, n_samples=10):
        self.n_words = n_words
        self.word_len = word_len
        self.n_samples = n_samples
    
    def get_random_font(self):
        fonts = [
                    cv2.FONT_HERSHEY_SIMPLEX,
#                    cv2.FONT_HERSHEY_PLAIN,
#                    cv2.FONT_HERSHEY_DUPLEX,
                    cv2.FONT_HERSHEY_COMPLEX,
                    cv2.FONT_HERSHEY_TRIPLEX,
#                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                    cv2.FONT_ITALIC
                ]
        font = np.random.choice(fonts)
        return font
    
    def get_random_bg(self, width = 128, height = 32):
        b, g, r = np.random.randint(0, 255, 3)
        color = (r, g, b)
        bg = np.zeros((height, width, 3), np.uint8)
        bg[:] = color
        return bg
    
    def create_image(self, word):
        bg = self.get_random_bg()
        h,w = bg.shape[:2]
        
        font = self.get_random_font()
            
        b, g, r = np.random.randint(0, 255, 3)
        color = (int(r), int(g), int(b))
        
        img = cv2.putText(bg, word, (5, 22), font, 0.8, color, 1, cv2.LINE_AA)    
        return img
    
    def get_words(self):
        alphabets = list(string.ascii_lowercase)
        words = [''.join(np.random.choice(alphabets, self.word_len)) for i in range(self.n_words)]
        return words
    
    def create_dataset(self):
        
        imgs = []
        labels = []
        
        words = self.get_words()
        
        for word in words:
            word_imgs = [self.create_image(word) for i in range(self.n_samples)]
            word_labels = [word] * self.n_samples
            imgs.extend(word_imgs)
            labels.extend(word_labels)
            
        return imgs, labels

        
        




