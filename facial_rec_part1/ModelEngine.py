import numpy as np
from ImageDataGenerator import DataGenerator
from DataReader import DataReader
import tensorflow as tf 

class ModelEngine():

              
        def CreateModel(self, output_dimension, input_dimension, lossType):

                #model
                model = tf.keras.Sequential([tf.keras.layers.Conv2D(128, kernel_size=3, activation=tf.nn.relu, input_shape=input_dimension),
                            tf.keras.layers.Conv2D(256, kernel_size=5, activation=tf.nn.relu),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                            
                            tf.keras.layers.Conv2D(256, kernel_size=3, activation=tf.nn.relu),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                            tf.keras.layers.Conv2D(256, kernel_size=5, activation=tf.nn.relu),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                            tf.keras.layers.Conv2D(256, kernel_size=5, activation=tf.nn.relu),
                            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(output_dimension, activation=tf.nn.softmax)])

                model.compile(optimizer='adam', 
                              loss=lossType,
                              metrics=['accuracy'])

                return model

        def fit_with_save(self, model, training_generator, validation_generator, numberOfEpochs=5, checkpointPath="training_1/cp.ckpt", saveWeightsOnly=True):

                checkpoint_path = checkpointPath

                cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=saveWeightsOnly,
                                                 verbose=0)

                model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=numberOfEpochs,
                    callbacks=[cp_callback],
                    use_multiprocessing=True,
                    workers=6)

        def new_from_existing(self, old_model, weights_path, num_classes, lossType, activationType):

                layers = old_model.layers[-3].output 
                model2 = tf.keras.Model(input=old_model.get_input_at(0), output=[layers])

                model2.load_weights(weights_path, by_name=True)
                
                second_layers = model2.layers[-1].output

                second_layers = tf.keras.layers.Flatten()(second_layers) 
                second_layers = tf.keras.layers.Dense(num_classes, activation=activationType)

                model3 = tf.keras.Model(input=model2.get_input_at(0), output=[second_layers])

                model3.compile(optimizer='adam', 
                              loss=lossType,
                              metrics=['accuracy'])

                return model3 
                

