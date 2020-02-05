import cv2
import dlib
import os

class LandmarkDetector():

    def __init__(self, predictor_file_path):
        self.Detector = dlib.get_frontal_face_detector()
        self.Predictor = dlib.shape_predictor(predictor_file_path)

    def get_landmarks(self, image):

        landmarks = []

        #convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #get the faces (assume only one), and get the predicted landmarks
        faces = self.Detector(image, 0)

        for face in faces: 

            predicted_landmarks = self.Predictor(image, face)

            #transfer the landmarks into a list
            for index in range(predicted_landmarks.num_parts):
                landmarks.append((predicted_landmarks.part(index).x, predicted_landmarks.part(index).y))

        return landmarks