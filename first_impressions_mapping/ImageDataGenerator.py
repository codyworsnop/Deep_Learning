import numpy as np
from tensorflow import keras
import cv2
import tensorflow as tf
import random
from ImageAugmentor import ImageAugmentor

class DataGenerator(keras.utils.Sequence):

    def __init__(self, image_names, labels, modelSettings, hogDetails=None, lbpDetails=None, augment=False, flattenImage=False):
        self.dimension = modelSettings['dimension'] 
        self.batch_size = modelSettings['batch_size']
        self.labels = labels
        self.image_names = image_names
        self.n_channels = modelSettings['n_channels']
        self.n_classes = modelSettings['n_classes']
        self.shuffle = modelSettings['shuffle']
        self.labelDataType = modelSettings['labelDataType']
        self.weight_path = modelSettings['weight_path']
        self.num_epochs = modelSettings['number_of_epochs']
        self.on_epoch_end()
        self.flatten = flattenImage

        self.Augmentor = ImageAugmentor(augment, hogDetails, lbpDetails)


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if (self.shuffle == True):
            np.random.shuffle(self.indexes)


    def __data_generation(self, image_names):
            
        x_images = []
        y_labels = [] 

        # Generate data
        for index, image_name in enumerate(image_names):

            #AM15DIHR
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)

            if (image is not None):
                image = cv2.resize(image, dsize=(self.dimension[1], self.dimension[0]), interpolation=cv2.INTER_CUBIC)

                augmented_image = self.Augmentor.Augment(image)

                #only care about the augmented image
                if (augmented_image is not None):
                    image = augmented_image

                if (self.flatten):
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = image.flatten() 

                x_images.append(image)
                y_labels.append(self.labels[image_name])

        X = np.asarray(x_images)   
        y = np.asarray(y_labels)   

        return X, y

    def __len__(self):

        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_names) / self.batch_size))

    def __getitem__(self, index, shouldReturnPaths = False):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        # Find list of IDs
        image_names_temp = [self.image_names[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_names_temp)

        if (shouldReturnPaths):
            return X, y, image_names_temp
        
        return X, y

    def binary_balance(self, y_batch): 

        target_distribution = 0.50

        totals = np.sum(y_batch, axis=0)
        distribution = totals / len(y_batch)
        weights = np.zeros(y_batch.shape)

        for index, value in enumerate(distribution):

            #distributiddons are equal, go to next label
            if (value == target_distribution):
                continue

            #if the value is over represented in the batch
            if (value > target_distribution):
                
                self.binary_balance_helper(target_distribution, y_batch, index, value, weights, 1, 0)

            #if the value is under represented in the batch 
            elif (value < target_distribution):

                self.binary_balance_helper(target_distribution, y_batch, index, value, weights, 0, 1)

        return weights 


    def binary_balance_helper(self, target_distribution, batch_labels, distribution_index, distribution_value, weights, over_represented_label, under_represented_label):

        # 1. Give a random subset of target_distribution 1, else 0 

        #Indices of over represented attributes
        indices = list(np.where(batch_labels[:,distribution_index] == over_represented_label)[0])
        indicesToUpdate = random.sample(indices, int(target_distribution * len(batch_labels)))

        for index in indicesToUpdate:
            weights[index][distribution_index] = 1

        # 2. Give the under represented samples target_distribution / batch_distribution weights 

        #Indices of negative examples
        indices = np.where(batch_labels[:,distribution_index] == under_represented_label)[0]

        #The batch density of the under represented attributes
        batch_density = len(indices) / len(batch_labels)

        #update each under represented label to have a higher weight. Pt(a) / Pb(a) 
        for index in indices:  
            weights[index][distribution_index] = (target_distribution / batch_density)
        

    def ShowImage(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        
    def plot_images(self, images):

        image = images[0]
        for nextimg in images:
            image = np.concatenate((image, nextimg), axis=1)

        self.ShowImage(image)