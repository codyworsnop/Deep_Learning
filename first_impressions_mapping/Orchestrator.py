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
from ModelType import ModelType 
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
        self.Metrics = Metrics()
    
    def SetupAndTrain(self, model, generator, features, settings, labels, kdef=False, validation_gen=None, test_gen=None): 

            validation_accuracy = None
            losses = Losses(10, settings)

            if (kdef):

                cost = losses.Mean_Squared_Error(label=labels, pred=model)
                accuracy = self.Metrics.kdef_accuracy(labels, model, settings[self.ParameterConstants.BatchSize]) 
                validation_accuracy = self.Metrics.MAPE(labels, model) 

            else:
                cost = losses.reduce_mean(labels, model)
                accuracy = Metrics().celeba_accuracy(labels, model)

            self.modelEngine.fit_with_save(features, labels, accuracy, generator, validation_gen, test_gen, cost, settings, model, kdef, validation_accuracy)

            return model

    def Run(self):

        features = tf.placeholder(tf.float32, shape=[None, self.modelSettings.celeba_params[self.ParameterConstants.Dimension][0],
                                                                self.modelSettings.celeba_params[self.ParameterConstants.Dimension][1], 
                                                                self.modelSettings.celeba_params[self.ParameterConstants.NumberOfChannels]], name='features')
        #create new model 
        model = self.modelEngine.CreateModel(features, self.modelSettings.celeba_params, ModelType.MobileNetV2)

         #train celeb_a if no weights exist for it 
        if (not self.reader.weights_exist(self.modelSettings.celeba_params['weight_path'] + '.meta')):

            #train celeb_a
            (celeba_partition, celeba_labels) = self.reader.read_celeb_a()
            labels = tf.placeholder(tf.float32, [None, self.modelSettings.celeba_params[self.ParameterConstants.NumberOfClasses]], name="celeba_predictions")

            #[:int(len(celeba_partition['train']) * .10)]
            training_gen = DataGenerator(celeba_partition['train'], celeba_labels, self.modelSettings.celeba_params, True)
            validation_gen = DataGenerator(celeba_partition['validation'], celeba_labels, self.modelSettings.celeba_params, True)
            test_gen = DataGenerator(celeba_partition['test'], celeba_labels, self.modelSettings.celeba_params, False)
            self.SetupAndTrain(model, training_gen, features, self.modelSettings.celeba_params, labels, validation_gen=validation_gen, test_gen=test_gen)
    
        #train kdef
        if (not self.reader.weights_exist(self.modelSettings.kdef_params[self.ParameterConstants.WeightPath] + '.meta')):

            (kdef_partition, kdef_labels) = self.reader.read_kdef()
            labels = tf.placeholder(tf.float32, [None, self.modelSettings.kdef_params[self.ParameterConstants.NumberOfClasses]], name="kdef_predictions")
            model = self.modelEngine.new_from_existing(model, self.modelSettings.kdef_params)
            training_gen = DataGenerator(kdef_partition['train'] + kdef_partition['validation'], kdef_labels, self.modelSettings.kdef_params, False)

            #validation_gen = DataGenerator(kdef_partition['validation'], kdef_labels, self.modelSettings.kdef_params)
            test_gen = DataGenerator(kdef_partition['test'], kdef_labels, self.modelSettings.kdef_params, False)

            #Dont pass in validation gen on KDEF. we need data. 
            model = self.SetupAndTrain(model, training_gen, features, self.modelSettings.kdef_params, labels, True, None, test_gen)    

orchestrator = LogProxy(Orchestrator())
orchestrator.Run()
