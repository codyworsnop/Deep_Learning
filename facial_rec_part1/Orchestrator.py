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



tf.enable_eager_execution()

#constants
weight_checkpoint_path = "./weights.h5"

Logger = Logger('./logs/', 'log')

#read kdef / celeb a
reader = LogProxy(DataReader())
modelEngine = ModelEngine()

#create new model 
model = modelEngine.CreateModel(settings.celeba_params)

#train celeb_a if no weights exist for it 
if (not reader.weights_exist(weight_checkpoint_path)):

    #read celeb_a
    (celeba_partition, celeba_labels) = reader.read_celeb_a()

    #celeb a generators
    #[:int(len(celeba_partition['train']) * .10)]
    celeba_training_generator = DataGenerator(celeba_partition['train'], celeba_labels, settings.celeba_params)
    celeba_validation_generator = DataGenerator(celeba_partition['validation'], celeba_labels, settings.celeba_params)

    #train model
    modelEngine.fit_with_save(model, celeba_training_generator, celeba_validation_generator, settings.celeba_params)

#read kdef
(kdef_partition, kdef_labels) = reader.read_kdef()

#kdef generators
training_generator = DataGenerator(kdef_partition['train'], kdef_labels, settings.kdef_params)
validation_generator = DataGenerator(kdef_partition['validation'], kdef_labels, settings.kdef_params)

kdef_metrics = Metrics(validation_generator, settings.kdef_params)

kdef_model = modelEngine.new_from_existing(model, settings.kdef_params, weight_checkpoint_path)
modelEngine.fit_with_save(kdef_model, training_generator, validation_generator, settings.kdef_params, callbacks=[kdef_metrics])

#test model on kdef data, lower learning rate with adam, 10e-5 learning rate, accuracy write own 

