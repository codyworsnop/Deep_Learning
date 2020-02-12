from interface import Interface

class IModel(Interface):
    '''The interface definition for models ''' 

    def Train(self, train_features, train_labels, validation_features, validation_labels):
        ''' Trains a given model on features and labels '''
        pass 

    def Predict(self, features, labels):
        ''' Predicts on test data for a given model '''
        pass
