import numpy as np
from ImageDataGenerator import DataGenerator
from DataReader import DataReader
import tensorflow as tf 

class ModelEngine():
              
        def CreateModel(self, output_dimension, input_dimension, lossType):

                #model
                model = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, input_shape=input_dimension),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                            
                        tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

                           # tf.keras.layers.Conv2D(64, kernel_size=5, activation=tf.nn.relu),
                           # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                            #tf.keras.layers.Conv2D(64, kernel_size=5, activation=tf.nn.relu),
                            #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                        tf.keras.layers.Dropout(rate=0.4),
                        tf.keras.layers.Dense(output_dimension, activation=tf.nn.sigmoid)])

                model.compile(optimizer='adam', 
                              loss=lossType,
                              metrics=['accuracy'])

                return model

        def fit_with_save(self, model, training_generator, validation_generator, numberOfEpochs=1, checkpointPath="weights.hdf5", saveWeightsOnly=True):

                checkpoint_path = checkpointPath

                cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=saveWeightsOnly,
                                                 period=1,
                                                 verbose=1)

                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', 
                                                                      histogram_freq=0, 
                                                                      batch_size=32, 
                                                                      write_graph=True, 
                                                                      write_grads=False, 
                                                                      write_images=False, 
                                                                      embeddings_freq=0, 
                                                                      embeddings_layer_names=None, 
                                                                      embeddings_metadata=None, 
                                                                      embeddings_data=None, 
                                                                      update_freq='epoch')

                model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=numberOfEpochs,
                    callbacks=[cp_callback, tensorboard_callback])

        def new_from_existing(self, old_model, weights_path, num_classes, lossType, activationType):

                layers = old_model.layers[-3].output 
                model2 = tf.keras.Model(inputs=old_model.get_input_at(0), outputs=[layers])
                
                model2.load_weights(weights_path, by_name=True)
                second_layers = model2.layers[-1].output

                second_layers = tf.keras.layers.Flatten()(second_layers) 
                second_layers = tf.keras.layers.Dense(num_classes, activation=activationType)(second_layers)

                model3 = tf.keras.Model(inputs=model2.get_input_at(0), outputs=[second_layers])

                model3.compile(optimizer='adam', 
                              loss=lossType,
                              metrics=['accuracy'])

                return model3 
                

