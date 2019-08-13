import numpy as np
from ImageDataGenerator import DataGenerator
from DataReader import DataReader
import tensorflow as tf 
import multiprocessing
from Losses import Losses as custom_losses
from ModelSettings import ModelParameterConstants
from ModelSettings import ModelParameters
from Metrics import Metrics
from tensorflow import keras
from Logger import Logger
import ApplicationConstants

#some code reused from https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
class ModelEngine():
              
        def __init__(self):
                self.ParameterConstants = ModelParameterConstants() 
                self.GlobalModelSettings = ModelParameters()           
                self.Saveables = []
                self.Logger = Logger(ApplicationConstants.LoggingFilePath, ApplicationConstants.LoggingFileName)
        def CreateModel(self, features, modelSettings):

                features = tf.reshape(features, shape=[-1, modelSettings[self.ParameterConstants.Dimension][0],
                                                                   modelSettings[self.ParameterConstants.Dimension][1], 
                                                                   modelSettings[self.ParameterConstants.NumberOfChannels]])                                                   

                layer1 = self.create_conv_layer(features, modelSettings[self.ParameterConstants.NumberOfChannels], 32, [5, 5], [2, 2], name='layer1')
                layer2 = self.create_conv_layer(layer1, 32, 128, [5, 5], [2, 2], name='layer2')
                layer3 = self.create_conv_layer(layer2, 128, 256, [5, 5], [2, 2], name='layer3')
                layer4 = self.create_conv_layer(layer3, 256, 128, [5, 5], [2, 2], name='layer4')
                layer5 = self.create_conv_layer(layer4, 128, 64, [5, 5], [2, 2], name='layer5')

                self.outputConvLayer = layer5
         
                output_dimension = layer5.shape[1] * layer5.shape[2] * layer5.shape[3]
                flattened = tf.reshape(layer5, [-1, output_dimension.value])

                wd1 = tf.Variable(tf.truncated_normal([output_dimension.value, 1024], stddev=0.03), name='wd1')
                bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
                dense_layer1 = tf.matmul(flattened, wd1) + bd1
                dense_layer1 = tf.nn.relu(dense_layer1)
                dense_layer1 = tf.nn.dropout(dense_layer1, modelSettings[self.ParameterConstants.DropoutRate])

                wd2 = tf.Variable(tf.truncated_normal([1024, modelSettings[self.ParameterConstants.NumberOfClasses]], stddev=0.03), name='wd2')
                bd2 = tf.Variable(tf.truncated_normal([modelSettings[self.ParameterConstants.NumberOfClasses]], stddev=0.01), name='bd2')
                dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
                output = tf.nn.sigmoid(dense_layer2)

                return output

        def fit_with_save(self, x, y, accuracy, training_generator, cost, modelSettings, model, restore=False):

                #optimiser = tf.train.AdamOptimizer(learning_rate=modelSettings[self.ParameterConstants.LearningRate]).minimize(cost)
                optimiser = tf.train.RMSPropOptimizer(learning_rate=modelSettings[self.ParameterConstants.LearningRate]).minimize(cost)

                init_op = tf.global_variables_initializer()

                if (any(self.Saveables)):
                        self.Logger.Info("Saving variables: " + str(self.Saveables))
                        saver = tf.train.Saver(var_list=self.Saveables)
                
                with tf.Session() as sess:
                        
                        sess.run(init_op)

                        if (restore):
                                weight_path = self.GlobalModelSettings.celeba_params[self.ParameterConstants.WeightPath]
                                self.Logger.Info("Restoring model weights from " + str(weight_path))
                                saver.restore(sess, weight_path)

                        self.Logger.Info("Starting training with step size " + str(modelSettings[self.ParameterConstants.LearningRate]))
                        for epoch in range(modelSettings[self.ParameterConstants.NumberOfEpochs]):
                                
                                batches = len(training_generator)
                                epoch_avg_loss = 0

                                for batch in range(batches):
                                        batch_x, batch_y = training_generator.__getitem__(batch)
                                        _, batch_loss = sess.run([optimiser, cost], feed_dict={x: batch_x, y: batch_y})
                                        epoch_avg_loss += batch_loss
     
                                        print("On batch:", batch, "of", batches, "for epoch:", (epoch + 1), "of", modelSettings[self.ParameterConstants.NumberOfEpochs], 
                                        "accuracy: ", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))

                                print("Epoch:", (epoch + 1), "average cost =", "{:.3f}".format(epoch_avg_loss / batches), "\n")
                        
                        print("\nTraining complete!")
                        saver.save(sess,  modelSettings[self.ParameterConstants.WeightPath])     

        def test(self, x, y, model, accuracy, test_generator, settings):

                init_op = tf.global_variables_initializer()

                with tf.Session() as sess:
                        sess.run(init_op)
                        batches = len(test_generator)

                        for batch in range(batches):
                                batch_x, batch_y = test_generator.__getitem__(batch)
                                print(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
                
                print("Done testing")
                

        def new_from_existing(self, old_model, modelSettings):

                convOutput = self.outputConvLayer

                output_dimension = convOutput.shape[1] * convOutput.shape[2] * convOutput.shape[3]
                flattened = tf.reshape(convOutput, [-1, output_dimension.value])

                wd1 = tf.Variable(tf.truncated_normal([output_dimension.value, 1024], stddev=0.03), name='wd1')
                bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
                dense_layer1 = tf.matmul(flattened, wd1) + bd1
                dense_layer1 = tf.nn.relu(dense_layer1)
                dense_layer1 = tf.nn.dropout(dense_layer1, modelSettings[self.ParameterConstants.DropoutRate])

                wd2 = tf.Variable(tf.truncated_normal([1024, modelSettings[self.ParameterConstants.NumberOfClasses]], stddev=0.03), name='wd2')
                bd2 = tf.Variable(tf.truncated_normal([modelSettings[self.ParameterConstants.NumberOfClasses]], stddev=0.01), name='bd2')
                dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2

                #no activation
                output = dense_layer2 

                return output 

        def create_conv_layer(self, input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):

                # setup the filter input shape for tf.nn.conv_2d
                conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

                # initialise weights and bias for the filter
                weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
                bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

                self.Saveables.append(weights)
                self.Saveables.append(bias)

                # setup the convolutional layer operation
                out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    
                # add the bias
                out_layer += bias

                # apply a ReLU non-linear activation
                out_layer = tf.nn.relu(out_layer)

                # now perform max pooling
                ksize = [1, pool_shape[0], pool_shape[1], 1]
                strides = [1, 2, 2, 1]
                out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

                return out_layer
