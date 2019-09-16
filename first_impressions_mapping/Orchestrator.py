import numpy as np

from ImageDataGenerator import DataGenerator
from DataReader import DataReader
from ModelEngine import ModelEngine
from Proxy import Proxy as LogProxy
from Logger import Logger
from Metrics import Metrics
from DataAnalytics import DataAnalytics

from ModelSettings import ModelParameters
from ModelSettings import ModelParameterConstants

from Losses import Losses

from tensorflow import keras
import tensorflow as tf
import cv2

#tf.enable_eager_execution()

class Orchestrator():
    
    def __init__(self):
        self.reader = LogProxy(DataReader())
        self.modelEngine = ModelEngine()
        self.modelSettings = ModelParameters() 
        self.ParameterConstants = ModelParameterConstants()
    
    def SetupAndTrain(self, model, generator, features, settings, labels, kdef=False, validation_gen=None): 

            accuracy = None 

            #train model
            if (kdef):
                loss = Losses(20, settings)
                cost = loss.bucketized_MSE(label=labels, pred=model)
               # cost = tf.losses.mean_squared_error(labels=labels, predictions=model) 
                accuracy = Metrics().kdef_accuracy(labels, model, settings[self.ParameterConstants.BatchSize]) 

            else:
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=labels))
                accuracy = Metrics().celeba_accuracy(labels, model)

            self.modelEngine.fit_with_save(features, labels, accuracy, generator, validation_gen, cost, settings, model, kdef)

            return model

    def Run(self):

        features = tf.placeholder(tf.float32, shape=[None, self.modelSettings.celeba_params[self.ParameterConstants.Dimension][0],
                                                                self.modelSettings.celeba_params[self.ParameterConstants.Dimension][1], 
                                                                self.modelSettings.celeba_params[self.ParameterConstants.NumberOfChannels]], name='features')

        #create new model 
        model = self.modelEngine.CreateModel(features, self.modelSettings.celeba_params)

         #train celeb_a if no weights exist for it 
        if (not self.reader.weights_exist(self.modelSettings.celeba_params['weight_path'] + '.meta')):

            #train celeb_a
            (celeba_partition, celeba_labels) = self.reader.read_celeb_a()
            labels = tf.placeholder(tf.float32, [None, self.modelSettings.celeba_params[self.ParameterConstants.NumberOfClasses]], name="celeba_predictions")

            #[:int(len(celeba_partition['train']) * .10)]
            training_gen = DataGenerator(celeba_partition['train'], celeba_labels, self.modelSettings.celeba_params)
            self.SetupAndTrain(model, training_gen, features, self.modelSettings.celeba_params, labels)
    
        #train kdef
        (kdef_partition, kdef_labels) = self.reader.read_kdef()
        labels = tf.placeholder(tf.float32, [None, self.modelSettings.kdef_params[self.ParameterConstants.NumberOfClasses]], name="kdef_predictions")
        model = self.modelEngine.new_from_existing(model, self.modelSettings.kdef_params)

        if (not self.reader.weights_exist(self.modelSettings.kdef_params[self.ParameterConstants.WeightPath] + '.meta')):
            training_gen = DataGenerator(kdef_partition['train'], kdef_labels, self.modelSettings.kdef_params)
            validation_gen = DataGenerator(kdef_partition['validation'], kdef_labels, self.modelSettings.kdef_params)
            model = self.SetupAndTrain(model, training_gen, features, self.modelSettings.kdef_params, labels, True, validation_gen)    

        #test kdef
        test_generator = DataGenerator(kdef_partition['test'], kdef_labels, self.modelSettings.kdef_params)
        kdef_accuracy = Metrics().kdef_accuracy(labels, model, self.modelSettings.kdef_params[self.ParameterConstants.BatchSize]) 
        self.modelEngine.test(features, labels, model, kdef_accuracy, test_generator, self.modelSettings.kdef_params)

orchestrator = LogProxy(Orchestrator())
orchestrator.Run()

#try mworks2, 4