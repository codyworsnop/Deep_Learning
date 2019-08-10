import tensorflow as tf 
import numpy as np

trust_predictions = []

class Metrics():
    
    def kdef_accuracy(self, y, pred): 
            
            diff = tf.abs(tf.subtract(tf.abs(y), tf.abs(pred)))
            trust_diff_sum = tf.reduce_sum(diff, axis=0)

            return trust_diff_sum

    def celeba_accuracy(self, y, pred):
      
            correct_prediction = tf.equal(tf.round(pred), tf.round(y)) 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy 