from tensorflow import keras
import tensorflow as tf
from DataReader import DataReader

class Losess():

    def __init__(self, numberOfBuckets, settings):
        self.NumberOfBuckets = numberOfBuckets
        self.Min = settings['min']
        self.Max = settings['max']
        self.Buckets = self.create_buckets()

    def bucketized_MSE(self, label, pred):
        bucket_index_label = self.Get_bucket_index(label)
        bucket_index_pred = self.Get_bucket_index(pred)
        bucket_index_diff = abs(bucket_index_label - bucket_index_pred)

        return keras.backend.mean(keras.backend.square(pred - label), axis=-1) * bucket_index_diff

    def create_buckets(self):
        stepSize = (self.Max - self.Min) / self.NumberOfBuckets
        value = self.Min
        buckets = []

        for _ in range(self.NumberOfBuckets + 1):

            buckets.append(round(value, 2))
            value += stepSize

        return buckets
    
    def Get_bucket_index(self, value):

        for bucket_index, bucket_value in enumerate(self.Buckets):

            tf.print(value)
            if (value >= bucket_value and value < self.Buckets[bucket_index + 1]):
                return bucket_index

        return -1
