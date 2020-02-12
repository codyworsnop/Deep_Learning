from Interfaces.IModel import IModel
from interface import implements

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.neural_network import MLPRegressor

class NN(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_Model()

    def Build_Model(self):

        model = MLPRegressor()
        return model 

    def Train(self, train_features, train_labels, validation_features, validation_labels):
        self.Model.fit(train_features, train_labels)

    def Predict(self, features, labels): 
        return self.Model.predict(features), self.Model.score(features, labels)
