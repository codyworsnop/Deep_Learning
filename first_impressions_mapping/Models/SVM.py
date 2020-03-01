from Interfaces.IModel import IModel
from interface import implements
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from sklearn import svm

from Metrics import Metrics

class SVM(implements(IModel)):

    def __init__(self):
        self.Model = self.Build_SVM()
        
    def Build_SVM(self):

        svm_model = svm.SVR(kernel='rbf', gamma='auto')
        model = MultiOutputRegressor(svm_model)
        return model 

    def Train(self, train_features, train_labels, validation_features, validation_labels):
        self.Model.fit(train_features, train_labels)

    def Predict(self, features): 
        
        prediction = self.Model.predict(features) 

        return prediction