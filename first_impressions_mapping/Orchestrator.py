import numpy as np

from ImageDataGenerator import DataGenerator
from DataReader import DataReader
from ModelEngine import ModelEngine
from Proxy import Proxy as LogProxy
from Logger import Logger
from Metrics import Metrics
from DataAnalytics import DataAnalytics

import ModelSettings as modelSettings

from tensorflow import keras
import tensorflow as tf
import cv2

tf.enable_eager_execution()

class Orchestrator():
    
    def __init__(self):
        self.reader = LogProxy(DataReader())
        self.modelEngine = ModelEngine()

    def SetupAndTrain(self, model, features, labels, settings, shouldGenerateNew = False): 

            callbacks = []

            #celeb a generators
            #[:int(len(celeba_partition['train']) * .10)]
            training_gen = DataGenerator(features['train'], labels, settings)
            validation_gen = DataGenerator(features['validation'], labels, settings)

            #train model
            if (shouldGenerateNew):
                callbacks = [Metrics(validation_gen, settings)]
                model = self.modelEngine.new_from_existing(model, settings)

            self.modelEngine.fit_with_save(model, training_gen, validation_gen, settings, callbacks)

            return model

    def Run(self):

        #create new model 
        celeb_a_model = self.modelEngine.CreateModel(modelSettings.celeba_params)

         #train celeb_a if no weights exist for it 
        if (not self.reader.weights_exist(modelSettings.celeba_params['weight_path'])):

            #train celeb_a
            (celeba_partition, celeba_labels) = self.reader.read_celeb_a()
            self.SetupAndTrain(celeb_a_model, celeba_partition, celeba_labels, modelSettings.celeba_params)

        #train kdef
        (kdef_partition, kdef_labels) = self.reader.read_kdef()
        kdef_model = self.SetupAndTrain(celeb_a_model, kdef_partition, kdef_labels, modelSettings.kdef_params, True)

        self.Test(kdef_model, kdef_partition, kdef_labels, modelSettings.kdef_params)

    def Test(self, model, features, labels, settings):

        test_gen = DataGenerator(features['test'], labels, settings)     
        metrics = Metrics(test_gen, settings)
        mean = metrics.calculate_mean(model) 

        print(mean)
   
        

orchestrator = LogProxy(Orchestrator())
orchestrator.Run()