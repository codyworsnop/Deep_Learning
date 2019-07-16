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
        diff = label - pred
        bucket_diff = diff / stepSize

        return keras.backend.mean(keras.backend.square(pred - label), axis=-1) * tf.to_float(bucket_diff)
