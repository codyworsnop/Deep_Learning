
from tensorflow import keras

kdef_params = { 
                'dimension': (218, 178),
                'batch_size': 8, 
                'n_classes': 3,
                'n_channels': 3,
                'shuffle': True,
                'labelDataType': float,
                'lossType': keras.losses.mean_squared_error,
                'learningRate': 0.0005, 
                'output_activation': None,
                'weight_path': "./weights.h5",
                'number_of_epochs': 100,
                }

celeba_params = { 
                    'dimension': (218,178),
                    'batch_size': 64, 
                    'n_classes': 40,
                    'n_channels': 3,
                    'shuffle': True,
                    'labelDataType': bool,    
                    'lossType': keras.losses.binary_crossentropy,
                    'learningRate': 0.001, 
                    'output_activation': keras.activations.sigmoid,
                    'weight_path': "./weights.h5",
                    'number_of_epochs': 200,
                }