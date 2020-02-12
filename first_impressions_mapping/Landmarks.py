from Logger import Logger
import ApplicationConstants
from ImageDataGenerator import DataGenerator
from Models.NN import NN
from LandmarkDetector import LandmarkDetector
from Proxy import Proxy as LogProxy
from DataReader import DataReader
from Models.ModelSettings import ModelParameters
from Models.ModelSettings import ModelParameterConstants
import numpy as np 
import os.path
from os import path

class Landmarks():

    def __init__(self):
        self.landmark_detector = LandmarkDetector('first_impressions_mapping/shape_predict_68_face_landmarks.dat')
        self.reader = LogProxy(DataReader())
        self.modelSettings = ModelParameters() 
        self.Logger = Logger(ApplicationConstants.LoggingFilePath, ApplicationConstants.LoggingFileName)

    def get_landmarks(self, generator):

        all_landmarks = []
        all_labels = [] 
        invalid = 0
        for batch in range(len(generator)):

            self.Logger.Info("On batch: " + str(batch) + " of " + str(len(generator)))
 

            #get batches
            batch_x, batch_y, paths = generator.__getitem__(batch, True)

            for index, image in enumerate(batch_x): 

                label = batch_y[index]

                #get the landmarks for image
                landmarks = self.landmark_detector.get_landmarks(image, paths[index])

                if landmarks is not None and len(landmarks) > 1: 
                    all_landmarks.append(landmarks)
                    all_labels.append(label)
                else:
                    invalid += 1

        return all_landmarks, all_labels

    def run(self):

        #read 
        (split_partition, split_labels) = self.reader.read_kdef()

        #create generators
        training_gen = DataGenerator(split_partition['train'], split_labels, self.modelSettings.kdef_params)
        test_gen = DataGenerator(split_partition['test'], split_labels, self.modelSettings.kdef_params)

        #TODO: find best way to feed landmarks into model
        landmarks, labels = self.get_landmarks(training_gen) 
        test_landmarks, test_labels = self.get_landmarks(test_gen) 

        flattened_landmarks = []
        flattened_test_landmarks = [] 
        
        for landmark in landmarks: 
            flattened_landmarks.append([item for sublist in landmark for item in sublist])

        for landmark in test_landmarks: 
            flattened_test_landmarks.append([item for sublist in landmark for item in sublist])

        fl = np.asarray(flattened_landmarks)
        ftl = np.asarray(flattened_test_landmarks)

        #run against models 
        models = [NN()] #expand for SVM, KNN, whatever else

         for model in models:
            model.Train(fl, labels, None, None) #most scikit models don't have a partial fit, which is why we have to pass in all the data at once :( 
            prediction, score = model.Predict(ftl, test_labels)

        
if __name__ == "__main__":
    
    Landmarks().run()
