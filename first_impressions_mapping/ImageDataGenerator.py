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

    def Jeremy_balance(self, labels):
        
        labels = np.asarray(labels, dtype=int)
        self.attribute_count = labels.shape[1]
        self.P_T = np.ones(self.attribute_count) * 0.5
        
        mask = np.ones(labels.shape)
        sum_labs = np.sum(labels, 0)
        P_B = sum_labs / float(self.batch_size)
        weight_neg = (1 - self.P_T) / (1 - P_B)
        weight_neg[np.isinf(weight_neg)] = 1
        weight_pos = self.P_T / P_B
        weight_pos[np.isinf(weight_pos)] = 1

        for idx in range(self.attribute_count):
            new_lab = np.ones(self.batch_size)
            if P_B[idx] > self.P_T[idx]:
                pos_idxs = np.where(labels[:,idx] == 1)[0]
                np.random.shuffle(pos_idxs)
                num_keep = int(self.P_T[idx] * self.batch_size)
                one_pos_idxs = pos_idxs[0:num_keep]
                zeroed_pos_idxs = pos_idxs[num_keep:sum_labs[idx]]
                new_lab[zeroed_pos_idxs] = 0
                new_lab *= weight_neg[idx]
                new_lab[one_pos_idxs] = 1
                mask[:, idx] = new_lab

            elif P_B[idx] < self.P_T[idx]:
                neg_idxs = np.where(labels[:,idx] == 0)[0]
                np.random.shuffle(neg_idxs)
                num_neg = self.batch_size - sum_labs[idx]
                num_keep = int((1 - self.P_T[idx]) * self.batch_size)
                one_neg_idxs = neg_idxs[0:num_keep]
                zeroed_neg_idxs = neg_idxs[num_keep:num_neg]
                new_lab[zeroed_neg_idxs] = 0
                new_lab *= weight_pos[idx]
                new_lab[one_neg_idxs] = 1
                mask[:, idx] = new_lab

        return mask 

    def ShowImage(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        
    def plot_images(self, images):

        image = images[0]
        for nextimg in images:
            image = np.concatenate((image, nextimg), axis=1)

        self.ShowImage(image)