
import tensorflow as tf 
from tensorflow import keras

class ModelParameters():

    kdef_params = { 
                    'dimension': (218,178),
                    'batch_size': 8, 
                    'n_classes': 3,
                    'n_channels': 3,
                    'shuffle': True,
                    'labelDataType': "float32",
                    'lossType': tf.losses.mean_squared_error,
                    'learningRate': 0.0005,
                    'output_activation': None,
                    'weight_path': "./kdef_weights.ckpt",
                    'number_of_epochs': 10000,
                    'min' : 1.00,
                    'max' : 7.00,
                    'dropout_rate' : 0.5,
                    }

    celeba_params = { 
                        'dimension': (218,178),
                        'batch_size': 64, 
                        'n_classes': 40,
                        'n_channels': 3,
                        'shuffle': True,
                        'labelDataType': "float32",    
                        'lossType': keras.losses.binary_crossentropy,
                        'learningRate': 0.001, 
                        'output_activation': tf.nn.sigmoid,
                        'weight_path': "./celeba_weights.ckpt",
                        'number_of_epochs': 1,
                        'dropout_rate' : 0.5,
                    }

class ModelParameterConstants():
    Dimension = 'dimension'
    BatchSize = 'batch_size'
    NumberOfClasses = 'n_classes'
    NumberOfChannels = 'n_channels'
    ShouldShuffle = 'shuffle'
    LabelDataType = 'labelDataType'
    LossType = 'lossType'
    LearningRate = 'learningRate'
    OutputActivation = 'output_activation'
    WeightPath = 'weight_path'
    NumberOfEpochs = 'number_of_epochs'
    DatasetRangeMinimum = 'min'
    DatasetRangeMaximum = 'max'
    DropoutRate = 'dropout_rate'
