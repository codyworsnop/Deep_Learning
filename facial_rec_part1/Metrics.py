from tensorflow import keras
import numpy as np

trust_predictions = []

class Metrics(keras.callbacks.Callback):

    def __init__(self, val_data, batch_size = 8):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        print(self.validation_data)
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        
    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)
        total = batches * self.batch_size
        
        val_pred = np.zeros((total,1))
        val_true = np.zeros((total))
        X = np.empty((self.batch_size, *self.dimension, self.n_channels), dtype=np.uint8)
        y = np.empty((self.batch_size, self.n_classes), dtype=self.labelDataType)
        
        for batch in range(batches):
            xVal, yVal = self.validation_data.__getitem__(batch)
            val_pred[batch * self.batch_size : (batch+1) * self.batch_size] = np.asarray(self.model.predict(xVal)).round()
            val_true[batch * self.batch_size : (batch+1) * self.batch_size] = yVal
            
        val_pred = np.squeeze(val_pred)
     #   _val_f1 = f1_score(val_true, val_pred)
     #   _val_precision = precision_score(val_true, val_pred)
     #   _val_recall = recall_score(val_true, val_pred)
        
      #  self.val_f1s.append(_val_f1)
      #  self.val_recalls.append(_val_recall)
      #  self.val_precisions.append(_val_precision)
     
        
        return