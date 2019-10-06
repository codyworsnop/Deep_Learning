# Helper libraries
import numpy as np
import numpy as np
import glob
import os

from Logger import Logger

classes = { 'T':0, 'D':1, 'A':2 }
number_of_Classes = 3 
BaseDirectory = os.getcwd() 
SPLIT_LIST = ['./split_labels/kdef_split_one_fullPath.csv', './split_labels/kdef_split_two_fullPath.csv', './split_labels/kdef_split_three_fullPath.csv', './split_labels/kdef_split_four_fullPath.csv', './split_labels/kdef_split_five_fullPath.csv' ]

class DataReader():

    def __init__(self):
        self.Logger = Logger('./logs/', 'log')

    def find(self, name, path):
        for root, directory, files in os.walk(path):
            if name in files:
                return os.path.join(root, name), root
        return None, None
        
    def find_directory(self, name, path): 
        for root, _, files in os.walk(path):
            
            directory = root.split('/')[-1]

            if name == directory:
                return root
    
    def read_kdef(self):

        images = []
        labels = {}

        kdef_labels_path, _ = self.find('kdef_labels.csv', BaseDirectory)
        label_file = open(kdef_labels_path, 'r')
        lines = label_file.readlines()[1:]

        #find kdef image directory
        kdef_directory = self.find_directory('KDEF', BaseDirectory) 

        self.Logger.Info("Reading KDEF images from: " + str(kdef_directory))

        for line in lines:
            
            #split the csv
            imageName, _, _, _, _, trustworthiness, dominance, attractiveness = line.rstrip('\n').split(',')
            imageName = imageName.upper()
            imagePath, _ = self.find(imageName, kdef_directory)

            if (imagePath == None):
                self.Logger.Error("The image path was None while reading kdef data: " + str(imageName))
                continue
            
            images.append(imagePath)

            labels[imagePath] = float(trustworthiness), float(dominance), float(attractiveness)

        return ({ 'train' : images[:int(len(images) * 0.70)], 'validation' : images[int(len(images) * .71) : int(len(images) * .90)], 'test' : images[int(len(images) * .91) : int(len(images))]}, labels)

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
            line = line.strip().split(' ')[1:]

            #labels.append([int(x) for x in line[1:]])
            for label in line:
            
                if (label == '-1' or label == '-1\n'):
                    labels.append(0)      

                elif (label == '1' or label == '1\n'):
                    labels.append(1)

            labels_dict[imagePath] = labels
            labels = []

        return ({ 'train' : images[:int(len(images) * .70)], 'validation': images[int(len(images) * .71) : int(len(images) * .90)], 'test' : images[int(len(images) * 0.91) : int(len(images))] }, labels_dict)

    def Read_Splits(self, split_count):

        if (split_count >= len(SPLIT_LIST)):
            print("Split count outside of index range")
            return

        VALNAMES = []
        TRAINNAMES = []
        VALLABELS = []
        TRAINLABELS = []
        IMGNAMES = []
        LABELS = []

        labels_dict = {}

        for i in range(5):
            if SPLIT_LIST[i] != SPLIT_LIST[split_count]:
                f = open(SPLIT_LIST[i],'r')
                for line in f:
                    line = line.strip().split(',')
                    temp = line[0].replace('.jpg','.JPG')
                    IMGNAMES.append(temp)
                    LABELS.append(line[5])
                    LABELS.append(line[6])
                    LABELS.append(line[7])
                f.close

        LABELS = np.asarray(LABELS)
        LABELS = np.reshape(LABELS,(-1,3))

        for i in range(len(LABELS)):
            TRAINNAMES.append(IMGNAMES[i])
            TRAINLABELS.append(LABELS[i])

            labels_dict[IMGNAMES[i]] = LABELS[i]

        # Read in validation data
        IMGNAMES = []
        LABELS = []

        f = open(SPLIT_LIST[split_count],'r')
        for line in f:
            line = line.strip().split(',')
            temp = line[0].replace('.jpg','.JPG')
            IMGNAMES.append(temp)
            LABELS.append(line[5])
            LABELS.append(line[6])
            LABELS.append(line[7])
        f.close

        LABELS = np.asarray(LABELS)
        LABELS = np.reshape(LABELS,(-1,3))

        for i in range(len(LABELS)):
            VALNAMES.append(IMGNAMES[i])
            VALLABELS.append(LABELS[i])   

            labels_dict[IMGNAMES[i]] = LABELS[i] 

        return ({ 'train' : TRAINNAMES, 'test' : VALNAMES }, labels_dict )
    
    def Fix_Paths(self):

        f = open('./split_labels/kdef_split_five.csv','r')
        fout = open('./split_labels/kdef_split_five_fullPath.csv', 'w')

        for readline in f:
            line = readline.strip().split(',')
            imageName = line[0].replace('.jpg', '.JPG')
            fullPath, _ = self.find(imageName, './KDEF')

            newstring = readline.replace(line[0], fullPath)
            fout.write(newstring)

        f.close()
        fout.close()

    def weights_exist(self, file_path):
        return os.path.exists(file_path)
