# Helper libraries
import numpy as np
import numpy as np
import glob
import os

classes = { 'T':0, 'D':1, 'A':2 }
number_of_Classes = 3 
BaseDirectory = os.getcwd() 

class DataReader():

    def find(self, name, path):
        for root, directory, files in os.walk(path):
            if name in files:
                return os.path.join(root, name), root
        return None, None
        
    def find_directory(self, name, path): 
        for root, _, files in os.walk(path):
            if name in root:
                return root
    
    def read_kdef(self):

        images = []
        labels = {}

        kdef_labels_path, _ = self.find('kdef_labels.csv', BaseDirectory)
        label_file = open(kdef_labels_path, 'r')
        lines = label_file.readlines()[1:]

        for line in lines:
            
            #split the csv
            imageName, _, _, _, _, trustworthiness, dominance, attractiveness = line.rstrip('\n').split(',')
            imageName = imageName.upper()

            imagePath, _ = self.find(imageName, BaseDirectory + '/KDEF')
            images.append(imagePath)

            labels[imagePath] = float(trustworthiness), float(dominance), float(attractiveness)

        return ({ 'train' : images[:int(len(images) * 0.70)], 'validation' : images[int(len(images) * .71) : int(len(images) * .90)] }, labels)

    def read_celeb_a(self):

        images = []

        #load labels
        label_file_path, _ = self.find('list_attr_celeba.txt', BaseDirectory)
        label_file = open(label_file_path, 'r')
        labels_lines = label_file.readlines()[2:]

        labels = []
        labels_dict = {}

        for line in labels_lines:

            #find image path
            imagePath = BaseDirectory + '/img_align_celeba/' + line.split(' ', 1)[0]
            images.append(imagePath)

            #extract features
            line = line.split(' ', 1)[1]
            
            for label in line.split(' '):
            
                if (label == '-1' or label == '-1\n'):
                    labels.append(False)      

                elif (label == '1' or label == '1\n'):
                    labels.append(True)

            labels_dict[imagePath] = labels
            labels = []

        return ({ 'train' : images[:int(len(images) * .70)], 'validation': images[int(len(images) * .71) : int(len(images) * .90)] }, labels_dict)

    def weights_exist(self, file_path):
        return os.path.exists(file_path)

