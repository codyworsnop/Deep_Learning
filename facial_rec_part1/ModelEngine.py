import numpy as np
from ImageDataGenerator import DataGenerator
from DataReader import DataReader
import tensorflow as tf 

class ModelEngine():
              
        def CreateModel(self, modelSettings):

                model = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, input_shape=(modelSettings['dimension'][0], modelSettings['dimension'][1], modelSettings['n_channels'])),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                            
                        tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation=tf.nn.relu),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

                        tf.keras.layers.Conv2D(256, kernel_size=5, activation=tf.nn.relu),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                        tf.keras.layers.Conv2D(64, kernel_size=5, activation=tf.nn.relu),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                        tf.keras.layers.Dense(modelSettings['n_classes'], activation=modelSettings['output_activation'])])

                model.summary()
                model.compile(tf.keras.optimizers.Adam(lr=modelSettings['learningRate']), 
                              loss=modelSettings['lossType'],
                              metrics=['accuracy'])

                return model

        def fit_with_save(self, model, training_generator, validation_generator, modelSettings, numberOfEpochs=1, callbacks=[], checkpointPath="weights.h5", saveWeightsOnly=True):

                model.fit_generator(generator=training_generator,
                                    validation_data=validation_generator,
                                    epochs=numberOfEpochs,
                                    callbacks=callbacks,
                                    workers=6,
                                    use_multiprocessing=False)

                model.save_weights(checkpointPath)

        def new_from_existing(self, old_model, weights_path, num_classes, lossType, activationType, learningRate=0.001, metrics=['accuracy']):

                old_model.summary()
                layers = old_model.layers[-4].output 
                model2 = tf.keras.Model(inputs=old_model.get_input_at(0), outputs=[layers])
                model2.summary()
                
                model2.load_weights(weights_path, by_name=True)
                second_layers = model2.layers[-1].output
                
                second_layers = tf.keras.layers.Flatten()(second_layers) 
                second_layers = tf.keras.layers.Dense(num_classes, activation=activationType)(second_layers)

                model3 = tf.keras.Model(inputs=model2.get_input_at(0), outputs=[second_layers])
                model3.summary()

                model3.compile(tf.keras.optimizers.Adam(lr=learningRate),
                              loss=lossType,
                              metrics=['accuracy'])

                return model3 