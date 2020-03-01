from Logger import Logger
import ApplicationConstants
from ImageDataGenerator import DataGenerator
from Models.NN import NN
from Models.SVM import SVM
from LandmarkDetector import LandmarkDetector
from Proxy import Proxy as LogProxy
from DataReader import DataReader
from Models.ModelSettings import ModelParameters
from Models.ModelSettings import ModelParameterConstants
import numpy as np 
import os.path
from os import path
import cv2

class Landmarks():

    def __init__(self):
        self.landmark_detector = LandmarkDetector('shape_predict_68_face_landmarks.dat')
        self.reader = LogProxy(DataReader())
        self.modelSettings = ModelParameters() 
        self.Logger = Logger(ApplicationConstants.LoggingFilePath, ApplicationConstants.LoggingFileName)

    def get_landmarks(self, generator):

        all_landmarks = []
        all_labels = [] 
        invalid = 0

        image_stacks = None

        for batch in range(0, len(generator)):

            self.Logger.Info("On batch: " + str(batch) + " of " + str(len(generator)))

            #get batches
            batch_x, batch_y, paths = generator.__getitem__(batch, True)
            hori_stack = None
            row = 0 

            for index, image in enumerate(batch_x): 

                label = batch_y[index]

                #get the landmarks for image   
                landmarks = self.landmark_detector.get_landmarks(image, paths[index], False)
                # vi = self.landmark_detector.visualize_landmarks(image, landmarks)
        
                # if (hori_stack is None):
                #     hori_stack = vi 
                # else:
                #     hori_stack = np.concatenate((hori_stack, vi), axis=1) 

                if landmarks is not None and len(landmarks) > 1: 
                    all_landmarks.append(landmarks)
                    all_labels.append(label)

                else:
                    invalid += 1


            # if (image_stacks is None):
            #     image_stacks = hori_stack
            # else:
            #     image_stacks = np.concatenate((image_stacks, hori_stack), axis=0)


        print("Invalid count:", invalid)
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
        models = [SVM()] #expand for SVM, KNN, whatever else

        for model in models:
            model.Train(fl, labels, None, None) #most scikit models don't have a partial fit, which is why we have to pass in all the data at once :( 
            prediction = model.Predict(ftl)
            accuracy = self.kdef_accuracy(prediction, test_labels, len(test_labels))

            binned_accuracy_05 = self.kdef_accuracy_bin(prediction, test_labels, len(test_labels))
            binned_accuracy_1 = self.kdef_accuracy_bin(prediction, test_labels, len(test_labels), bin_diff=1.0)
            binned_accuracy_15 = self.kdef_accuracy_bin(prediction, test_labels, len(test_labels), bin_diff=1.5)

            print('accuracy:', accuracy)
            print('binned accuracy 0.5', binned_accuracy_05)
            print('binned accuracy 1.0', binned_accuracy_1)
            print('binned accuracy 1.5', binned_accuracy_15)


    def kdef_accuracy(self, y, pred, batch_size): 

        diff = np.abs(np.subtract(pred, y))
        trust_diff_sum = np.sum(diff, axis=0)

        return trust_diff_sum / batch_size

    def kdef_accuracy_bin(self, y, pred, batch_size, bin_diff = 0.5):
        
        bins = np.empty_like(pred, int)

        for prediction_index, prediction in enumerate(pred):
            label = y[prediction_index]

            for attribute_index, attribute_prediction in enumerate(prediction): 
                attribute_label = label[attribute_index]

                if (abs(attribute_label - attribute_prediction) <= bin_diff):
                    bins[prediction_index][attribute_index] = 1
                else:
                    bins[prediction_index][attribute_index] = 0

        sum_predictions = np.sum(bins, axis=0)        
        accuracy = sum_predictions / batch_size

        return accuracy

    def get_bin_index(self, value, min_value, max_value, bin_step): 

        min_value = 1
        max_value = 7 

        bin_step = (max_value - min_value) / num_bins
        bins = []

        for index, current in enumerate(range(min_value, max_value + 1, step=bin_step)): 
            if (value >= current and value <= current + bin_step):
                return index

        return -1 

if __name__ == "__main__":
    
    Landmarks().run()