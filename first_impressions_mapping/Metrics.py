import tensorflow as tf 
import numpy as np
from ModelSettings import ModelParameterConstants

trust_predictions = []

class Metrics():

    def __init__(self, numberOfBins, settings):
        self.NumberOfBins = numberOfBins
        self.Min = settings[ModelParameterConstants.DatasetRangeMinimum]
        self.Max = settings[ModelParameterConstants.DatasetRangeMaximum]

        if (numberOfBins != 0):
            bins = np.zeros(self.NumberOfBins + 1)

            stepSize = (self.Max - self.Min) / self.NumberOfBins
            current = self.Min
            for bin_index in range(0, len(bins)):
                bins[bin_index] = current
                current = current + stepSize
            
            self.Bins = bins
    
    def kdef_accuracy(self, y, pred, batch_size): 
      
        diff = tf.abs(tf.subtract(pred, y))
        trust_diff_sum = tf.reduce_sum(diff, axis=0)

        return trust_diff_sum / batch_size
    
    def kdef_accuracy_SVM(self, labels, prediction, batch_size):

        diff = np.absolute(np.subtract(prediction, labels))
        trust_diff_sum = np.sum(diff, axis=0)

        return trust_diff_sum / batch_size

    def MAPE(self, y, prediction):

        mean_absolute_percent_error = tf.abs(y - prediction)/y
        MAPE = tf.reduce_mean(mean_absolute_percent_error)

        return MAPE 

    def celeba_accuracy(self, y, pred):
      
        correct_prediction = tf.equal(tf.round(pred), tf.round(y)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy
        
    def kdef_thresholding(self, y, pred):

        stepSize = (self.Max - self.Min) / self.NumberOfBins
        shaped_stepSize = tf.fill(tf.shape(pred), stepSize)
        diff = tf.abs(pred - y)
        binary_bins = tf.cast(tf.less(diff, shaped_stepSize), tf.int32)
        summed_bins = tf.math.reduce_sum(binary_bins, axis=0) 
        return summed_bins / tf.shape(pred)[0]


    def kdef_binary_bin(self, y, pred):

        y_bins = np.zeros(y.shape)
        pred_bins = np.zeros(pred.shape)
        
        for x in range(0, y.shape[0]):
            for y_index in range(0, y.shape[1]):
                for bin_index in range(0, len(self.Bins)):
                    if (y[x][y_index] >= self.Bins[bin_index] and y[x][y_index] < self.Bins[bin_index + 1]):
                        y_bins[x][y_index] = bin_index
                        break

        for x in range(0, pred.shape[0]):
            for y_index in range(0, pred.shape[1]):
                for bin_index in range(0, len(self.Bins)):
                    if (pred[x][y_index] >= self.Bins[bin_index] and pred[x][y_index] < self.Bins[bin_index + 1]):
                        pred_bins[x][y_index] = bin_index
                        break

        correct_bins = (np.equal(y_bins, pred_bins)).astype(int)
        mean_correct_bins = np.mean(correct_bins, axis=0)
        return mean_correct_bins

    def calculate_mean(self, sess, model, training_generator, x):

        trust_diff_sum = 0
        batches = len(training_generator)

        for batch in range(batches):

                print ("validating kdef accuracy. Batch " + str(batch) + " of " + str(batches))

                batch_x, batch_y = training_generator.__getitem__(batch)
                pred = sess.run(model, feed_dict={x: batch_x})

                print ("yval: " + str(batch_y[0]))
                print ("pred: " + str(pred[0]))
                diff = np.abs(np.subtract(np.abs(batch_y), np.abs(pred)))
                trust_diff_sum = np.sum(diff, axis=0)

        mean = trust_diff_sum / batches

        return mean