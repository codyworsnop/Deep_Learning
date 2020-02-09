from Logger import Logger
import ApplicationConstants
from ImageDataGenerator import DataGenerator
from Models.NN import NN
from LandmarkDetector import LandmarkDetector
from Proxy import Proxy as LogProxy
from DataReader import DataReader
from Models.ModelSettings import ModelParameters
from Models.ModelSettings import ModelParameterConstants

class Landmarks():

    def __init__(self):
        self.landmark_detector = LandmarkDetector('shape_predict_68_face_landmarks.dat')
        self.reader = LogProxy(DataReader())
        self.modelSettings = ModelParameters() 
        self.Logger = Logger(ApplicationConstants.LoggingFilePath, ApplicationConstants.LoggingFileName)

    def get_landmarks(self, generator):

        all_landmarks = []

        for batch in range(len(generator)):

            self.Logger.Info("On batch: " + str(batch) + " of " + str(len(generator)))

            #get batches
            batch_x, batch_y = generator.__getitem__(batch)

            for index, image in enumerate(batch_x): 

                label = batch_y[index]

                #get the landmarks for image
                landmarks = self.landmark_detector.get_landmarks(image)
                all_landmarks.append((landmarks, label))

        return all_landmarks

    def run(self):

        #read 
        (split_partition, split_labels) = self.reader.read_kdef()

        #create generators
        training_gen = DataGenerator(split_partition['train'], split_labels, self.modelSettings.kdef_params)
        test_gen = DataGenerator(split_partition['test'], split_labels, self.modelSettings.kdef_params)

        #TODO: find best way to feed landmarks into model
        landmarks = self.get_landmarks(training_gen) 

        #run against models 
        models = [NN()] #expand for SVM, KNN, whatever else

        for model in models:
            model.Train(landmarks) #most scikit models don't have a partial fit, which is why we have to pass in all the data at once :( 

if __name__ == "__main__":
    
    Landmarks().run()
