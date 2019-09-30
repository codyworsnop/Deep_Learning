import tensorflow as tf 
import numpy as np

trust_predictions = []

class Metrics():
    
    def kdef_accuracy(self, y, pred, batch_size): 
            
            diff = tf.abs(tf.subtract(pred, y))
            trust_diff_sum = tf.reduce_sum(diff, axis=0)

            return trust_diff_sum / batch_size

    def MAPE(self, y, prediction):

            mean_absolute_percent_error = tf.abs(y - prediction)/y
            MAPE = tf.reduce_mean(mean_absolute_percent_error)

            return MAPE 

    def celeba_accuracy(self, y, pred):
      
            correct_prediction = tf.equal(tf.round(pred), tf.round(y)) 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy

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