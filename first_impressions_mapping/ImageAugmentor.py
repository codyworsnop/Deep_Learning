
import tensorflow as tf
import math
import numpy as np
import cv2
import random
from skimage.feature import hog
from skimage import data, exposure
from skimage.feature import local_binary_pattern
import os
import numpy
import PIL 
from PIL import Image 

class ImageAugmentor(): 

    def __init__(self, shouldRunGenericAugmentations, hogDetails, lbpDetails):
        self.shouldAugmentGeneric = shouldRunGenericAugmentations
        self.HogDetails = hogDetails
        self.LbpDetails = lbpDetails
    
    def Augment(self, image):

        images = []
        
        #Augment the image
        if (self.shouldAugmentGeneric):
            augmented = self.__AugmentImage(image) 
            images.append(augmented)

        #self.ShowImage(image)
        if (self.HogDetails is not None):

            hog_images = [] 
            for split_image in self.__splitChannels(image):
                _, hog_result_image = hog(split_image, orientations=self.HogDetails.Orientations, pixels_per_cell=self.HogDetails.PixelsPerCell, cells_per_block=self.HogDetails.CellsPerBlock, visualize= self.HogDetails.Visualize, multichannel=False)
                hog_images.append(hog_result_image)

            averaged_image = self.__average_image(hog_images)

            if (self.HogDetails.ShouldFlatten):
                averaged_image = averaged_image.flatten()

            images.append(averaged_image)

        if (self.LbpDetails is not None):

            lbp_images = [] 

            for split_image in self.__splitChannels(image):
                lbp = local_binary_pattern(split_image, self.LbpDetails.NumberOfPoints, self.LbpDetails.Radius)
                lbp_images.append(lbp) 

            averaged_image = self.__average_image(lbp_images)
            
            if (self.LbpDetails.ShouldFlatten):
                averaged_image = averaged_image.flatten() 

            images.append(averaged_image)

        return images

    def ShowImage(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def __splitChannels(self, image):
        red = image[:,:,2]
        green = image[:,:,1]
        blue = image[:,:,0]

        return red, green, blue

    def __RotateImage(self, image):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, random.randint(-45, 45), 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def __Flip(self, image):
        flipped_img = np.fliplr(image)
        return flipped_img

    def __Translate(self, image):
        trans_range = 80
        rows, cols, ch = image.shape   
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        return cv2.warpAffine(image,Trans_M,(cols,rows))

    def __average_image(self, images):

        N = len(images)
        averaged_image = numpy.zeros(images[0].shape, numpy.float)

        for image in images: 
            averaged_image += image / N

        return averaged_image 

    def __AugmentImage(self, image):

        augmented = self.Translate(image) 
        augmented = self.RotateImage(augmented)    
        #augmented = self.Flip(augmented) #data already has both sides of face, so no need to flip horizontally 

        return augmented