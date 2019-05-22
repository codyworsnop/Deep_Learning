import numpy as np

from ImageDataGenerator import DataGenerator
from DataReader import DataReader
from ModelEngine import ModelEngine
from Proxy import Proxy as LogProxy

import ModelSettings as settings
from tensorflow import keras
import tensorflow as tf

#constants
weight_checkpoint_path = "weights.hdf5"

#read kdef / celeb a
reader = LogProxy(DataReader())
modelEngine = LogProxy(ModelEngine())

#create new model 
model = modelEngine.CreateModel(output_dimension=40, input_dimension=(218, 178, 3), lossType=keras.losses.binary_crossentropy)

#train celeb_a if no weights exist for it 
if (not reader.weights_exist(weight_checkpoint_path)):

    #read celeb_a
    (celeba_partition, celeba_labels) = reader.read_celeb_a()

    #celeb a generators
    #[:int(len(celeba_partition['train']) * .10)]
    celeba_training_generator = DataGenerator(celeba_partition['train'][:int(len(celeba_partition['train']) * .01)], celeba_labels, **settings.celeba_params)
    celeba_validation_generator = DataGenerator(celeba_partition['validation'], celeba_labels, **settings.celeba_params)

    #train model
    modelEngine.fit_with_save(model, celeba_training_generator, None, checkpointPath=weight_checkpoint_path)

#read kdef
(kdef_partition, kdef_labels) = reader.read_kdef()

#kdef generators
training_generator = DataGenerator(kdef_partition['train'], kdef_labels, **settings.kdef_params)
validation_generator = DataGenerator(kdef_partition['validation'], kdef_labels, **settings.kdef_params)

kdef_model = modelEngine.new_from_existing(model, weight_checkpoint_path, 3, keras.losses.mean_squared_error, keras.activations.sigmoid)
kdef_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=1)


