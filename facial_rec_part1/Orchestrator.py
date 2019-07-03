import numpy as np

from ImageDataGenerator import DataGenerator
from DataReader import DataReader
from ModelEngine import ModelEngine
from Proxy import Proxy as LogProxy
from Logger import Logger
from Metrics import Metrics

import ModelSettings as settings
from tensorflow import keras
import tensorflow as tf
from Metrics import Metrics
import cv2

class Orchestrator():
    
    def __init__(self):
        self.reader = LogProxy(DataReader())
        self.modelEngine = ModelEngine()

        self.weight_checkpoint_path = "./weights.h5"

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

    def Run(self):

        #create new model 
        celeb_a_model = self.modelEngine.CreateModel(settings.celeba_params)

         #train celeb_a if no weights exist for it 
        if (not self.reader.weights_exist(self.weight_checkpoint_path)):

            #train celeb_a
            (celeba_partition, celeba_labels) = self.reader.read_celeb_a()
            self.SetupAndTrain(celeb_a_model, celeba_partition, celeba_labels, settings.celeba_params)

        #train kdef
        (kdef_partition, kdef_labels) = self.reader.read_kdef()
        self.SetupAndTrain(celeb_a_model, kdef_partition, kdef_labels, settings.kdef_params, True)


orchestrator = Orchestrator()
orchestrator.Run()