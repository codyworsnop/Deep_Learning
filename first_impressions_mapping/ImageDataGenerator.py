import numpy as np
from tensorflow import keras
import cv2
from DataAugmenter import DataAugmenter
import tensorflow as tf
import random

class DataGenerator(keras.utils.Sequence):

    def __init__(self, image_names, labels, modelSettings, augment):
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
        self.DataAugmentation = DataAugmenter()

        #data needs to have its own class. Move out label data type, shuffle, augment data, etc into a "GenericData" class
        self.AugmentData = augment


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if (self.shuffle == True):
            np.random.shuffle(self.indexes)


    def __data_generation(self, image_names):

        # Initialization
        if (self.AugmentData):
            batchSize = self.batch_size * 2
        else:
            batchSize = self.batch_size

        X = np.empty((batchSize, *self.dimension, self.n_channels), dtype="uint8")
        y = np.empty((batchSize, self.n_classes), dtype=self.labelDataType)

        # Generate data
        for index, image_name in enumerate(image_names):

            #AM15DIHR
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)

            if (image is not None):
                #image = np.resize(image, (self.dimension[0], self.dimension[1], self.n_channels))
                image = cv2.resize(image, dsize=(self.dimension[1], self.dimension[0]), interpolation=cv2.INTER_CUBIC)

                #Augment the image
                if (self.AugmentData):

                    augmented = self.DataAugmentation.augmentImage(image) 
                    X[batchSize - index - 1] = augmented
                    y[batchSize - index - 1] = self.labels[image_name]
 
                X[index] = image
                y[index] = self.labels[image_name]           

        return X, y

    def __len__(self):

        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_names) / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        # Find list of IDs
        image_names_temp = [self.image_names[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_names_temp)

        #self.ShowImage(X[0])
        return X, y

    def binary_balance(self, y_batch):


        test = np.array([[1, 1, 0],
                            [0, 1, 0],
                            [1, 1, 0], 
                            [0, 1, 0], 
                            [1, 1, 0], 
                            [0, 1, 0], 
                            [1, 1, 0], 
                            [0, 0, 1], 
                            [1, 0, 1], 
                            [0, 0, 1]])

        y_batch = test 

        target_distribution = 0.50

        totals = np.sum(y_batch, axis=0)
        distribution = totals / len(y_batch)
        weights = np.zeros(y_batch.shape)

        for index, value in enumerate(distribution):

            #distributions are equal, go to next label
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

        # 1. give a random subset of target_distribution 1, else 0 

        #indices of postive samples
        indices = list(np.where(batch_labels[:,distribution_index] == over_represented_label)[0])
        indicesToUpdate = random.sample(indices, int(target_distribution * len(batch_labels)))

        for index in indicesToUpdate:
            weights[index][distribution_index] = 1

        # 2. give the under represented samples target_distribution / batch_distribution weights 
        indices = np.where(batch_labels[:,distribution_index] == under_represented_label)[0]
        batch_density = len(indices) / len(batch_labels)
        
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