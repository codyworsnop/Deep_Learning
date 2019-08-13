from tensorflow import keras
import tensorflow as tf
from DataReader import DataReader

class Losses():

    def __init__(self, numberOfBuckets, settings):
        self.NumberOfBuckets = numberOfBuckets
        self.Min = settings['min']
        self.Max = settings['max']

    def bucketized_MSE(self, label, pred):

        stepSize = (self.Max - self.Min) / self.NumberOfBuckets
        diff = tf.abs(pred - label)
        bucket_diff = diff / stepSize
        meaned_bucket_diff = tf.abs(keras.backend.mean(bucket_diff, axis=-1))
        result = tf.reduce_mean(tf.square(pred - label), axis=-1) * tf.to_float(meaned_bucket_diff)
        return result

#try mean, then sum 