from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from Metrics import Metrics
from Logger import Logger
from Models.ModelSettings import ModelParameterConstants
import ApplicationConstants
import numpy as np
import cv2

class SvmEngine():
    """ The engine used for any SVM model """ 
    
    def __init__(self, settings, kernel='rbf'):
        
        self.Kernel = kernel
        self.Metrics = Metrics(0, settings)
        self.Logger = Logger(ApplicationConstants.LoggingFilePath, ApplicationConstants.LoggingFileName)
        self.Settings = settings

    def Build_SVM(self):
        """ creates an SVR SVM for regression """ 

        svm_model = svm.SVR(self.Kernel, gamma='auto', verbose=True) 
        self.Model = MultiOutputRegressor(svm_model)    

    def Fit(self, training_gen):

        batches = len(training_gen)
        batch_x_all, batch_y_all = training_gen.__getitem__(0)

        for batch in range(1, batches):

            self.Logger.Info("On batch: " + str(batch) + " of " + str(batches))

            #get batches
            batch_x, batch_y = training_gen.__getitem__(batch)

            batch_x_all = np.concatenate((batch_x_all, batch_x), axis=0)
            batch_y_all = np.concatenate((batch_y_all, batch_y), axis=0)

        #fit model 
        self.Logger.Info("Fitting")
        self.Model.fit(batch_x_all, batch_y_all)

        #get prediction for batch
        self.Logger.Info("Predicting")
        prediction = self.Model.predict(batch_x_all)

        #get accuracy
        accuracy = self.Metrics.kdef_accuracy_SVM(batch_y_all, prediction, len(training_gen) * self.Settings[ModelParameterConstants.BatchSize])

        self.Logger.Info('training accuracy: ' + str(accuracy))

        return accuracy

    def Predict(self, test_gen):
	
        batches = len(test_gen)
        batch_x_all, batch_y_all = test_gen.__getitem__(0)

        for batch in range(1, batches):

            print("gathering batch " + str(batch) + " of " + str(batches))

            self.Logger.Info("On batch: " + str(batch) + " of " + str(batches))

            batch_x, batch_y = test_gen.__getitem__(batch)

            batch_x_all = np.concatenate((batch_x_all, batch_x), axis=0)
            batch_y_all = np.concatenate((batch_y_all, batch_y), axis=0)

        self.Logger.Info("Predicting")
        prediction = self.Model.predict(batch_x_all)

        accuracy = self.Metrics.kdef_accuracy_SVM(batch_y_all, prediction, len(test_gen) * self.Settings[ModelParameterConstants.BatchSize])

        self.Logger.Info('test accuracy: ' + str(accuracy))

        return accuracy
    
    def ShowImage(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)



        