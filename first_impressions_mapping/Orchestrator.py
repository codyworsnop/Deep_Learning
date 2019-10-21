import numpy as np

from ImageDataGenerator import DataGenerator
from DataReader import DataReader
from ModelEngine import ModelEngine
from Proxy import Proxy as LogProxy
from Logger import Logger
from Metrics import Metrics
from DataAnalytics import DataAnalytics
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from ModelSettings import ModelParameters
from ModelSettings import ModelParameterConstants
from Logger import Logger
from Losses import Losses
from ModelType import ModelType 
from tensorflow import keras
import ApplicationConstants
import tensorflow as tf
import cv2
#from SVMEngine import Svm

#tf.enable_eager_execution()

class Orchestrator():
    
    def __init__(self):
        self.reader = LogProxy(DataReader())
        self.modelEngine = ModelEngine()
        self.modelSettings = ModelParameters() 
        self.Logger = Logger(ApplicationConstants.LoggingFilePath, ApplicationConstants.LoggingFileName)
    
    def SetupAndTrain(self, model, generator, features, settings, labels, kdef=False, validation_gen=None, test_gen=None, save=True): 

            validation_accuracy = None
            losses = Losses(10, settings)

            if (kdef):
                metrics = Metrics(6, settings)
                cost = losses.Mean_Squared_Error(label=labels, pred=model)
                accuracy = metrics.kdef_accuracy(labels, model, settings[ModelParameterConstants.BatchSize]) 
                validation_accuracy = metrics.MAPE(labels, model) 
                label_weights = None

            else:
                label_weights = tf.placeholder(tf.float32, shape=labels.shape)
                cost = losses.reduce_mean2(labels, model, label_weights)
                accuracy = Metrics(6, settings).celeba_accuracy(labels, model)

            return self.modelEngine.fit_with_save(features, labels, accuracy, generator, validation_gen, test_gen, cost, settings, model, kdef, validation_accuracy, save, label_weights)

            #return model

    def Run_DL(self, cross_validate):

        if (cross_validate):
            split_totals = [] 

            for split in range(5):
                
                self.Logger.Info("\n\nBeginner " + str(split) + " split of k-fold\n\n")
                (split_partition, split_labels) = self.reader.Read_Splits(split)
                training_gen = DataGenerator(split_partition['train'], split_labels, self.modelSettings.kdef_params, False)
                test_gen = DataGenerator(split_partition['test'], split_labels, self.modelSettings.kdef_params, False)

                result = self.Train_KDEF(training_gen, test_gen, save=False)
                split_totals.append(result) 

                #cleanup 
                tf.reset_default_graph()
                self.modelEngine.Saveables = [] 
            
            for index, split in enumerate(split_totals):
                print('split ' + str(index) + ' accuracy is: trust ' +  str(split[0]) + ', domiance: ' + str(split[1]) + ", attraction: " + str(split[2]))

        else:

            (kdef_partition, kdef_labels) = self.reader.read_kdef()
            training_gen = DataGenerator(kdef_partition['train'] + kdef_partition['validation'], kdef_labels, self.modelSettings.kdef_params, False)

            #validation_gen = DataGenerator(kdef_partition['validation'], kdef_labels, self.modelSettings.kdef_params)
            test_gen = DataGenerator(kdef_partition['test'], kdef_labels, self.modelSettings.kdef_params, False)

            self.Train_KDEF(training_gen, test_gen)
            

    def Train_KDEF(self, training_generator, test_generator, save=True): 

        #defining weights for the celeba model since we're transfer learning potentially. 
        features = tf.placeholder(tf.float32, shape=[None, self.modelSettings.celeba_params[ModelParameterConstants.Dimension][0],
                                                           self.modelSettings.celeba_params[ModelParameterConstants.Dimension][1], 
                                                           self.modelSettings.celeba_params[ModelParameterConstants.NumberOfChannels]], 
                                                           name='features_celeba') 

        #train celeb a first for transfer learning
        model = self.Train_CelebA(features)

        #train kdef
        if (not self.reader.weights_exist(self.modelSettings.kdef_params[ModelParameterConstants.WeightPath] + '.meta')):
           
            labels = tf.placeholder(tf.float32, [None, self.modelSettings.kdef_params[ModelParameterConstants.NumberOfClasses]], name="kdef_predictions")
            model = self.modelEngine.new_from_existing(model, self.modelSettings.kdef_params)

            #Dont pass in validation gen on KDEF. we need data. 
            result = self.SetupAndTrain(model, training_generator, features, self.modelSettings.kdef_params, labels, True, None, test_generator, save=save)   

        return result 

    
    def Train_CelebA(self, features):

        #create new model 
        model = self.modelEngine.CreateModel(features, self.modelSettings.celeba_params, ModelType.MobileNetV2)

         #train celeb_a if no weights exist for it 
        if (not self.reader.weights_exist(self.modelSettings.celeba_params['weight_path'] + '.meta')):

            #train celeb_a
            (celeba_partition, celeba_labels) = self.reader.read_celeb_a()
            labels = tf.placeholder(tf.float32, [None, self.modelSettings.celeba_params[ModelParameterConstants.NumberOfClasses]], name="celeba_predictions")

            #[:int(len(celeba_partition['train']) * .10)]
            training_gen = DataGenerator(celeba_partition['train'], celeba_labels, self.modelSettings.celeba_params, False)
            validation_gen = DataGenerator(celeba_partition['validation'], celeba_labels, self.modelSettings.celeba_params, False)
            test_gen = DataGenerator(celeba_partition['test'], celeba_labels, self.modelSettings.celeba_params, False)
            self.SetupAndTrain(model, training_gen, features, self.modelSettings.celeba_params, labels, validation_gen=None, test_gen=test_gen)
        
        return model 

    def Run_SVM(self):
        (kdef_partition, kdef_labels) = self.reader.read_kdef()
        image_file_path = kdef_partition['train'][0] 
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image)

        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
    
        plt.show()

    def ShowImage(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)


orchestrator = LogProxy(Orchestrator())
orchestrator.Run_SVM()
