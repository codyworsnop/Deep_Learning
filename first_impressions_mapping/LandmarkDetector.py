import cv2
import dlib
import os
import subprocess
import glob

import pandas as pd
from natsort import natsorted

class LandmarkDetector():

    def __init__(self, predictor_file_path):
        self.Detector = dlib.get_frontal_face_detector()
        self.Predictor = dlib.shape_predictor(predictor_file_path)

    def visualize_landmarks(self, image, landmarks):

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        return image 

    def get_landmarks(self, image, image_path, useOpenFace = True):

        landmarks = []

        #convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #get the faces (assume only one), and get the predicted landmarks
        faces = self.Detector(image, 0)  

        if any(faces):

            for face in faces: 

                predicted_landmarks = self.Predictor(image, face)

                #transfer the landmarks into a list
                for index in range(predicted_landmarks.num_parts):
                    landmarks.append((predicted_landmarks.part(index).x, predicted_landmarks.part(index).y))

        else: 

            if (useOpenFace):
                #reset output directory 
                landmark_output_directory = 'first_impressions_mapping/OpenFace_landmarks/'

                files = glob.glob(landmark_output_directory + '*')
                for f in files:
                    os.remove(f)

                #Nates code here 
                OpenFaceBashCommand = '/home/codyworsnop/Documents/OpenFace/build/bin/FaceLandmarkImg -2Dfp -wild -f ' + image_path + ' -out_dir ' + landmark_output_directory + ''
                subprocess.call(OpenFaceBashCommand.split())

                # usecols parameter for opening OpenFace landmarks csv's in pandas dataframe
                un_landmarked_images_csv_usecols = ["face", "x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9",
                                                    "x_10", "x_11", "x_12", "x_13", "x_14", "x_15", "x_16", "x_17", "x_18", "x_19",
                                                    "x_20", "x_21", "x_22", "x_23", "x_24", "x_25", "x_26", "x_27", "x_28", "x_29",
                                                    "x_30", "x_31", "x_32", "x_33", "x_34", "x_35", "x_36", "x_37", "x_38", "x_39",
                                                    "x_40", "x_41", "x_42", "x_43", "x_44", "x_45", "x_46", "x_47", "x_48", "x_49",
                                                    "x_50", "x_51", "x_52", "x_53", "x_54", "x_55", "x_56", "x_57", "x_58", "x_59",
                                                    "x_60", "x_61", "x_62", "x_63", "x_64", "x_65", "x_66", "x_67", "y_0", "y_1", "y_2",
                                                    "y_3", "y_4", "y_5", "y_6", "y_7", "y_8", "y_9", "y_10", "y_11", "y_12", "y_13",
                                                    "y_14", "y_15", "y_16", "y_17", "y_18", "y_19", "y_20", "y_21", "y_22", "y_23",
                                                    "y_24", "y_25", "y_26", "y_27", "y_28", "y_29", "y_30", "y_31", "y_32", "y_33",
                                                    "y_34", "y_35", "y_36", "y_37", "y_38", "y_39", "y_40", "y_41", "y_42", "y_43",
                                                    "y_44", "y_45", "y_46", "y_47", "y_48", "y_49", "y_50", "y_51", "y_52", "y_53",
                                                    "y_54", "y_55", "y_56", "y_57", "y_58", "y_59", "y_60", "y_61", "y_62", "y_63",
                                                    "y_64", "y_65", "y_66", "y_67"]

                file_names = natsorted(os.listdir(landmark_output_directory))

                for file_name_index in range(len(file_names)):

                    if file_names[file_name_index][-4:] == ".csv":

                            # open the OpenFace landmarks in a pandas dataframe
                            OpenFace_landmarks = pd.read_csv(
                                landmark_output_directory + "/" + file_names[file_name_index],
                                sep=",",
                                usecols=un_landmarked_images_csv_usecols,
                                skipinitialspace=True)

                            OpenFace_landmarks.set_index('face', inplace=True)

                            # check if there are 0, 1, or more faces detected and handle the image respectively
                            row = 0 #guranteed to only have 1 face

                            for column in range(0, 68):

                                # extract the landmark coordinates from pandas data frame, round them, and convert them to ints
                                landmarks.append((int(round(OpenFace_landmarks.iloc[row][column])),
                                                int(round(OpenFace_landmarks.iloc[row][column + 68]))))

        return landmarks

