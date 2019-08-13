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
    
    def SetupAndTrain(self, model, generator, features, settings, kdef=False): 

            accuracy = None 
            labels = tf.placeholder(tf.float32, [None, settings[self.ParameterConstants.NumberOfClasses]], name="predictions")

            #train model
            if (kdef):
                model = self.modelEngine.new_from_existing(model, settings)

                loss = Losses(20, settings)

               # cost = loss.bucketized_MSE(label=labels, pred=model)
                cost = tf.losses.mean_squared_error(labels=labels, predictions=model) 
                accuracy = Metrics().kdef_accuracy(labels, model, settings[self.ParameterConstants.BatchSize]) 

            else:
                cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=labels))
                accuracy = Metrics().celeba_accuracy(labels, model)

            self.modelEngine.fit_with_save(features, labels, accuracy, generator, cost, settings, model, kdef)

            return model

    def Run(self):

        features = tf.placeholder(tf.float32, shape=[None, self.modelSettings.celeba_params[self.ParameterConstants.Dimension][0],
                                                                self.modelSettings.celeba_params[self.ParameterConstants.Dimension][1], 
                                                                self.modelSettings.celeba_params[self.ParameterConstants.NumberOfChannels]], name='features')

        #create new model 
        celeb_a_model = self.modelEngine.CreateModel(features, self.modelSettings.celeba_params)

         #train celeb_a if no weights exist for it 
        if (not self.reader.weights_exist(self.modelSettings.celeba_params['weight_path'] + '.meta')):

            #train celeb_a
            (celeba_partition, celeba_labels) = self.reader.read_celeb_a()

            #[:int(len(celeba_partition['train']) * .10)]
            training_gen = DataGenerator(celeba_partition['train'], celeba_labels, self.modelSettings.celeba_params)
            self.SetupAndTrain(celeb_a_model, training_gen, features, self.modelSettings.celeba_params)
    
        #train kdef
        (kdef_partition, kdef_labels) = self.reader.read_kdef()
        training_gen = DataGenerator(kdef_partition['train'], kdef_labels, self.modelSettings.kdef_params)
        kdef_model = self.SetupAndTrain(celeb_a_model, training_gen, features, self.modelSettings.kdef_params, True)

    def Test(self, model, features, labels, settings):

        test_gen = DataGenerator(features['test'], labels, settings)     

        mean = Metrics().calculate_mean(model) 

        print(mean)      

orchestrator = LogProxy(Orchestrator())
orchestrator.Run()


#try mworks2, 4