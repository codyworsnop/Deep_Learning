import tensorflow as tf
import math
import numpy as np
import cv2

import random

class DataAugmenter():

    def RotateImage(self, image):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, random.randint(-45, 45), 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def Flip(self, image):
        flipped_img = np.fliplr(image)
        return flipped_img

    def Translate(self, image):
        trans_range = 80
        rows, cols, ch = image.shape   
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        return cv2.warpAffine(image,Trans_M,(cols,rows))

    def augmentImage(self, image):

        augmented = self.Translate(image) 
        augmented = self.RotateImage(augmented)    
        #augmented = self.Flip(augmented) #data already has both sides of face, so no need to flip horizontally 

        return augmented




