from tensorflow import keras
import numpy as np

trust_predictions = []

class Metrics(keras.callbacks.Callback):

    def __init__(self, val_data, modelSettings):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = modelSettings['batch_size']
        self.labelDataType = modelSettings['labelDataType']
        self.n_classes = modelSettings['n_classes']

    def on_train_begin(self, logs={}):
        print(self.validation_data)
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        
    def on_epoch_end(self, epoch, logs={}):
        
        mean = self.calculate_mean(self.model)

        print ('\nTrust mean diff: ' + str(mean[0]))
        print ('Dominance mean diff: ' + str(mean[1]))
        print ('Attractiveness mean diff: ' + str(mean[2]))
     
    def calculate_mean(self, model):
        batches = len(self.validation_data)
        trust_diff_sum = 0
      
        for batch in range(batches):

            print ("validating kdef accuracy. Batch " + str(batch) + " of " + str(batches))

            x_val, y_val = self.validation_data.__getitem__(batch)
            pred = model.predict(x_val)

            print ("yval: " + str(y_val[0]))
            print ("pred: " + str(pred[0]))
            diff = np.abs(np.subtract(np.abs(y_val), np.abs(pred)))
            trust_diff_sum = np.sum(diff, axis=0)

        mean = trust_diff_sum / batches

        return mean