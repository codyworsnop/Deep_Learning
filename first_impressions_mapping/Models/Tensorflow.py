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
from ModelSettings import ModelParameterConstants
import ApplicationConstants
import time
from tensorflow import keras 
import cv2
from ModelType import ModelType
import sys 

#some code reused from https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/

weight_decay=1e-4

class ModelEngine():
              
        def __init__(self):
                self.ParameterConstants = ModelParameterConstants() 
                self.GlobalModelSettings = ModelParameters()           
                self.Saveables = []
                self.Learnable_weights = [] 
                self.Logger = Logger(ApplicationConstants.LoggingFilePath, ApplicationConstants.LoggingFileName)
         
                                
        def CreateModel(self, features, modelSettings, modelType):

                if (modelType == ModelType.Basic):
                        return self.Basic_Model(features, modelSettings)
                elif (modelType == ModelType.MobileNetV2):
                        return self.CreateModel_mobileNet(features, modelSettings[ModelParameterConstants.NumberOfClasses])


        def fit_with_save(self, x, y, accuracy, training_generator, validation_gen, test_generator, cost, modelSettings, prediction, isKdef=False, validation_accuracy=None, save=True, label_weights=None):

                optimiser = tf.train.AdamOptimizer(learning_rate=modelSettings[self.ParameterConstants.LearningRate]).minimize(cost)
                epoch_count_with_higher_accuracy = 0
                lowest_accuracy = sys.maxsize

                if (validation_accuracy is None):
                        validation_accuracy = accuracy
                
                if (any(self.Saveables)):
                        celeba_saver = tf.train.Saver(var_list=self.Saveables)

                with tf.Session() as sess:

                        if (isKdef):
                                metrics = Metrics(6, modelSettings)
                                bin_accuracy = metrics.kdef_thresholding(y, prediction)
                                
                                kdef_saver = tf.train.Saver()
                                weight_path = self.GlobalModelSettings.celeba_params[self.ParameterConstants.WeightPath]
                                self.Logger.Info("Restoring model weights from " + str(weight_path))
                                celeba_saver.restore(sess, weight_path)
                        
                        #init variables
                        sess.run(tf.global_variables_initializer())

                        for epoch in range(modelSettings[self.ParameterConstants.NumberOfEpochs]):
                                
                                batches = len(training_generator)
                                epoch_avg_loss = 0

                                for batch in range(batches):
                                        batch_x, batch_y = training_generator.__getitem__(batch)

                                        if (label_weights is not None):
                                                balance_weights = training_generator.binary_balance(batch_y) 
                                                _, batch_loss = sess.run([optimiser, cost], feed_dict={x: batch_x, y: batch_y, label_weights: balance_weights})
                                        else:
                                                _, batch_loss = sess.run([optimiser, cost], feed_dict={x: batch_x, y: batch_y})   

                                        epoch_avg_loss += batch_loss

                                        batch_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                                        self.Logger.Info("On batch: " + str(batch) + " of " + str(batches) + " for epoch: " + str((epoch + 1)) + " of " + str(modelSettings[self.ParameterConstants.NumberOfEpochs]) +
                                        " accuracy: " + str(batch_accuracy))

                                
                                        #bin accuracy 
                                        if (isKdef):
                                                pred = sess.run(prediction, feed_dict={x: batch_x})
                                                bin_acc = metrics.kdef_binary_bin(batch_y, pred)
                                                bin_accuracy_result = sess.run(bin_accuracy, feed_dict={x: batch_x, y: batch_y})
                                                print("threshold:", str(bin_accuracy_result))
                                                print("correct:", str(bin_acc))

                                        #self.Logger.Info("y: " + str(batch_y))
                                        #pred = sess.run(prediction, feed_dict={x: batch_x})
                                        #self.Logger.Info('pred: ' + str(pred))
                                        #diff = tf.abs(tf.subtract(pred, y))
                                        #trust_diff_sum = tf.reduce_sum(diff, axis=0)
                                        #print(str(sess.run(trust_diff_sum, feed_dict={x: batch_x, y:batch_y})))

                                #run validation for early stopping condition evaluation
                                (lowest_accuracy, stop) = self.training_validation(sess, validation_gen, x, y, epoch, validation_accuracy, epoch_count_with_higher_accuracy, lowest_accuracy)
                                if (stop):
                                        break 

                        self.Logger.Info("\n******\nTraining complete!\n******\n\nStarting Testing\n")

                        if (save):
                                        
                                if (isKdef):
                                        kdef_saver.save(sess, modelSettings[self.ParameterConstants.WeightPath])                       
                                        
                                else:
                                        celeba_saver.save(sess, modelSettings[self.ParameterConstants.WeightPath])  
  
                        return self.training_test(sess, test_generator, accuracy, bin_accuracy, x, y, prediction, isKdef, modelSettings)


        def training_validation(self, sess, validation_gen, x, y, epoch, accuracy, epoch_count_with_higher_accuracy, lowest_accuracy):

                if (validation_gen is not None):
                        
                        shouldStop = False
                        validation_batches = len(validation_gen)
                        validation_accuracy = 0 
                        for batch in range(validation_batches):

                                batch_x, batch_y = validation_gen.__getitem__(validation_batches)
                                validation_accuracy += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

                                self.Logger.Info("Validating on batch: " + str(batch) + " of " + str(validation_batches) + 'validation accuracy: ' + str(validation_accuracy))
                        
                        #save off first mape value

                        if (validation_accuracy < lowest_accuracy):
                                self.Logger.Info('saving new lowest mape value')
                                lowest_accuracy = validation_accuracy
                                epoch_count_with_higher_accuracy = 0
                        else:
                                self.Logger.Info('incrementing higher mape count')
                                epoch_count_with_higher_accuracy += 1 

                        #stop training if validation error is getting worse after n epochs
                        if (epoch_count_with_higher_accuracy > 5):
                                self.Logger.Warn("Stopping learning after " + str(epoch) + " epochs with a MAPE value of " + str(validation_accuracy))
                                shouldStop = True

                        self.Logger.Info('\nvalidation total mean absolute percent error: ' + str(validation_accuracy))
                
                        return (lowest_accuracy, shouldStop) 
                
                return (lowest_accuracy, False)

        def training_test(self, sess, test_generator, accuracy, bin_accuracy, x, y, prediction, isKdef, settings):

                total_accuracy = 0
                total_bin_accuracy = 0
                total_correct_bin_accuracy = 0
                test_batches = len(test_generator)
                metrics = Metrics(6, settings)

                for batch in range(test_batches):

                        batch_x, batch_y = test_generator.__getitem__(batch)
                        batch_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                        bin_accuracy_result = sess.run(bin_accuracy, feed_dict={x: batch_x, y: batch_y})

                        pred = sess.run(prediction, feed_dict={x: batch_x})
                        bin_correct_accuracy = metrics.kdef_binary_bin(batch_y, pred)

                        self.Logger.Info("Testing on batch: " + str(batch) + " of " + str(test_batches) + " accuracy: " + str(batch_accuracy))
                        #pred = sess.run(prediction, feed_dict={x: batch_x})
                        #dif = sess.run(tf.abs(tf.subtract(batch_y, pred)))
                        #print("Difference", dif)
                        #print("summation", sess.run(tf.reduce_sum(dif, axis=0)))
                        total_bin_accuracy += bin_accuracy_result
                        total_accuracy += batch_accuracy
                        total_correct_bin_accuracy += bin_correct_accuracy

                total = total_accuracy / test_batches
                total2 = total_bin_accuracy / test_batches
                total3 = total_correct_bin_accuracy / test_batches

                if (isKdef):
                        self.Logger.Info('Total accuracy is: trust ' +  str(total[0]) + ', domiance: ' + str(total[1]) + ", attraction: " + str(total[2]))
                        self.Logger.Info('Total bin accuracy: trust ' +  str(total2[0]) + ', domiance: ' + str(total2[1]) + ", attraction: " + str(total2[2]))
                        self.Logger.Info('Total correct bin accuracy: trust ' +  str(total3[0]) + ', domiance: ' + str(total3[1]) + ", attraction: " + str(total3[2]))

                self.Logger.Info("\nDone testing")    
                
                return total

        def test(self, x, y, prediction, accuracy, test_generator, modelSettings):

                saver = tf.train.Saver(self.Saveables)

                with tf.Session() as sess:

                        weight_path = modelSettings[self.ParameterConstants.WeightPath]
                        self.Logger.Info("Restoring model weights from " + str(weight_path))
                        saver.restore(sess, weight_path)

                        sess.run(tf.global_variables_initializer())
                        batches = len(test_generator)

                        for batch in range(batches):

                                batch_x, batch_y = test_generator.__getitem__(batch)

                                print("On batch:", batch, "of", batches, "accuracy: ", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))

                                #print("Actual:", batch_y)
                                #pred = sess.run(prediction, feed_dict={x: batch_x})
                                #print("Prediction", pred)
                                #dif = sess.run(tf.abs(tf.subtract(batch_y, pred)))
                                #print("Difference", dif)
                                #print("summation", sess.run(tf.reduce_sum(dif, axis=0)))
                
                print("Done testing")
                

        def new_from_existing(self, old_model, modelSettings):

                convOutput = self.outputConvLayer

                output_dimension = convOutput.shape[1] * convOutput.shape[2] * convOutput.shape[3]
                flattened = tf.reshape(convOutput, [-1, output_dimension.value])

                wd1 = tf.Variable(tf.truncated_normal([output_dimension.value, 1024], stddev=0.03), name='wd1')
                bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')

                dense_layer1 = tf.matmul(flattened, wd1) + bd1
                dense_layer1 = tf.nn.relu(dense_layer1)
                dense_layer1 = tf.nn.dropout(dense_layer1, rate=modelSettings[self.ParameterConstants.DropoutRate])

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

        def Basic_Model(self, features, modelSettings):

                features = tf.reshape(features, shape=[-1, modelSettings[self.ParameterConstants.Dimension][0],
                                                           modelSettings[self.ParameterConstants.Dimension][1], 
                                                           modelSettings[self.ParameterConstants.NumberOfChannels]])                                                   

                layer1 = self.create_conv_layer(features, modelSettings[self.ParameterConstants.NumberOfChannels], 64, [5, 5], [2, 2], name='layer1')
                layer2 = self.create_conv_layer(layer1, 64, 64, [5, 5], [2, 2], name='layer2')
                layer3 = self.create_conv_layer(layer2, 64, 128, [5, 5], [2, 2], name='layer3')
                layer4 = self.create_conv_layer(layer3, 128, 128, [5, 5], [2, 2], name='layer4')
                layer5 = self.create_conv_layer(layer4, 128, 256, [5, 5], [2, 2], name='layer5')
                layer6 = self.create_conv_layer(layer5, 256, 512, [5, 5], [2, 2], name='layer6')

                self.outputConvLayer = layer6
         
                output_dimension = self.outputConvLayer.shape[1] * self.outputConvLayer.shape[2] * self.outputConvLayer.shape[3]
                flattened = tf.reshape(self.outputConvLayer, [-1, output_dimension.value])

                wd1 = tf.Variable(tf.truncated_normal([output_dimension.value, 1024], stddev=0.03), name='wd1')
                bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
                self.Saveables.append(wd1)
                self.Saveables.append(bd1)

                fc = tf.nn.relu(tf.matmul(flattened, wd1) + bd1)
                fc = tf.nn.dropout(fc, rate=modelSettings[self.ParameterConstants.DropoutRate])

                w_out = tf.Variable(tf.truncated_normal([1024, modelSettings[self.ParameterConstants.NumberOfClasses]], stddev=0.03), name='wd1')
                b_out = tf.Variable(tf.truncated_normal([modelSettings[self.ParameterConstants.NumberOfClasses]], stddev=0.01), name='bd1')

                output = tf.nn.sigmoid(tf.matmul(fc, w_out) + b_out)

                return output
                
                
#################################################################################################################################
#This is a MobileNetV2 implementation taken from https://github.com/neuleaf/MobileNetV2
#################################################################################################################################

        def CreateModel_mobileNet(self, inputs, num_classes, is_train=True, reuse=False):
                exp = 6  # expansion ratio
                with tf.variable_scope('mobilenetv2'):
                        net = self.conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1')  # size/2

                        net = self.res_block(net, 1, 16, 1, is_train, name='res2_1')

                        net = self.res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
                        net = self.res_block(net, exp, 24, 1, is_train, name='res3_2')

                        net = self.res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
                        net = self.res_block(net, exp, 32, 1, is_train, name='res4_2')
                        net = self.res_block(net, exp, 32, 1, is_train, name='res4_3')

                        net = self.res_block(net, exp, 64, 2, is_train, name='res5_1')
                        net = self.res_block(net, exp, 64, 1, is_train, name='res5_2')
                        net = self.res_block(net, exp, 64, 1, is_train, name='res5_3')
                        net = self.res_block(net, exp, 64, 1, is_train, name='res5_4')

                        net = self.res_block(net, exp, 96, 1, is_train, name='res6_1')  # size/16
                        net = self.res_block(net, exp, 96, 1, is_train, name='res6_2')
                        net = self.res_block(net, exp, 96, 1, is_train, name='res6_3')

                        net = self.res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
                        net = self.res_block(net, exp, 160, 1, is_train, name='res7_2')
                        net = self.res_block(net, exp, 160, 1, is_train, name='res7_3')

                        net = self.res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

                        net = self.pwise_block(net, 1280, is_train, name='conv9_1')
                        net = self.global_avg(net)

                        self.outputConvLayer = net
                        
                        logits = self.flatten(self.conv_1x1(net, num_classes, name='logits'))

                        pred = tf.nn.sigmoid(logits, name='prob')
                        return pred
                        
        def relu(self, x, name='relu6'):
                return tf.nn.relu6(x, name)

        def batch_norm(self, x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
                return tf.layers.batch_normalization(x,
                                momentum=momentum,
                                epsilon=epsilon,
                                scale=True,
                                training=train,
                                name=name)

        def conv2d(self, input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
                with tf.variable_scope(name):
                        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
                conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

                self.Saveables.append(w)
                if bias:
                        biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
                        self.Saveables.append(biases)
                        conv = tf.nn.bias_add(conv, biases)

                return conv

        def conv2d_block(self, input, out_dim, k, s, is_train, name):
                with tf.name_scope(name), tf.variable_scope(name):
                        net = self.conv2d(input, out_dim, k, k, s, s, name='conv2d')
                        net = self.batch_norm(net, train=is_train, name='bn')
                        net = self.relu(net)
                        return net

        def conv_1x1(self, input, output_dim, name, bias=False):
                with tf.name_scope(name):
                        return self.conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

        def pwise_block(self, input, output_dim, is_train, name, bias=False):
                with tf.name_scope(name), tf.variable_scope(name):
                        out=self.conv_1x1(input, output_dim, bias=bias, name='pwb')
                        out=self.batch_norm(out, train=is_train, name='bn')
                        out=self.relu(out)
                        return out

        def dwise_conv(self, input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
                padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
                with tf.variable_scope(name):
                        in_channel=input.get_shape().as_list()[-1]
                        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))

                        self.Saveables.append(w)
                        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
                        if bias:
                                biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
                                self.Saveables.append(biases)
                                conv = tf.nn.bias_add(conv, biases)

                        return conv

        def res_block(self, input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
                with tf.name_scope(name), tf.variable_scope(name):
                        # pw
                        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
                        net = self.conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
                        net = self.batch_norm(net, train=is_train, name='pw_bn')
                        net = self.relu(net)
                        # dw
                        net = self.dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
                        net = self.batch_norm(net, train=is_train, name='dw_bn')
                        net = self.relu(net)
                        # pw & linear
                        net = self.conv_1x1(net, output_dim, name='pw_linear', bias=bias)
                        net = self.batch_norm(net, train=is_train, name='pw_linear_bn')

                        # element wise add, only for stride==1
                        if shortcut and stride == 1:
                                in_dim=int(input.get_shape().as_list()[-1])
                                if in_dim != output_dim:
                                        ins=self.conv_1x1(input, output_dim, name='ex_dim')
                                        net=ins+net
                                else:
                                        
                                        net=input+net

                        return net

        def separable_conv(self, input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
                with tf.name_scope(name), tf.variable_scope(name):
                        in_channel = input.get_shape().as_list()[-1]
                        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                initializer=tf.truncated_normal_initializer(stddev=0.02))

                        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
                        strides = [1,stride, stride,1]

                        conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
                        if bias:
                                biases = tf.get_variable('bias', [output_dim],initializer=tf.constant_initializer(0.0))
                                conv = tf.nn.bias_add(conv, biases)
                        return conv

        def global_avg(self, x):
                with tf.name_scope('global_avg'):
                        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
                        return net

        def flatten(self, x):
        #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
                return tf.contrib.layers.flatten(x)

        def pad2d(self, inputs, pad=(0, 0), mode='CONSTANT'):
                paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
                net = tf.pad(inputs, paddings, mode=mode)
                return net