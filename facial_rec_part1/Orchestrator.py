import numpy as np

from ImageDataGenerator import DataGenerator
from DataReader import DataReader
from ModelEngine import ModelEngine
import ModelSettings as settings
from tensorflow import keras

#constants - wow it's like 135
weight_checkpoint_path = "celeba_weights/weights.ckpt"

#read kdef / celeb a
reader = DataReader() 
modelEngine = ModelEngine()


#create new model 
model = modelEngine.CreateModel(output_dimension=40, input_dimension=(218, 178, 3), lossType=keras.losses.binary_crossentropy)

#train celeb_a if no weights exist for it 
if (not reader.weights_exist(weight_checkpoint_path)):

    #read celeb_a
    (celeba_partition, celeba_labels) = reader.read_kdef()

    #celeb a generators
    celeba_training_generator = DataGenerator(celeba_partition['train'], celeba_labels, **settings.celeba_params)
    celeba_validation_generator = DataGenerator(celeba_partition['validation'], celeba_labels, **settings.kdef_params)

    #train model
    modelEngine.fit_with_save(model, celeba_training_generator, None, checkpointPath=weight_checkpoint_path)

#read kdef
(kdef_partition, kdef_labels) = reader.read_kdef()

#kdef generators
training_generator = DataGenerator(kdef_partition['train'], kdef_labels, **settings.celeba_params)
validation_generator = DataGenerator(kdef_partition['validation'], kdef_labels, **settings.kdef_params)

kdef_model = modelEngine.new_from_existing(model, )




