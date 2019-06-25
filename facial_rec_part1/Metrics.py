from tensorflow import keras as K

class Metrics():

    def trust(self, y_true, y_pred):
        return y_true[0]

    def dom(self, y_true, y_pred):
        return y_pred[1]

    def attr(self, y_true, y_pred):
        return y_pred[2]

