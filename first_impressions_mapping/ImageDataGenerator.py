import numpy as np
from tensorflow import keras
import cv2

class DataGenerator(keras.utils.Sequence):

    def __init__(self, image_names, labels, modelSettings):
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


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if (self.shuffle == True):
            np.random.shuffle(self.indexes)


    def __data_generation(self, image_names):

        # Initialization
        X = np.empty((self.batch_size, *self.dimension, self.n_channels), dtype="float32")
        y = np.empty((self.batch_size, self.n_classes), dtype=self.labelDataType)

        # Generate data
        for index, image_name in enumerate(image_names):

            image = cv2.imread(image_name, cv2.IMREAD_COLOR)

            if (image is not None):
                X[index] = np.resize(image, (self.dimension[0], self.dimension[1], self.n_channels))
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

    #    self.ShowImage(X[0])
        return X, y