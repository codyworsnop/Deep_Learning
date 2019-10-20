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

    def Mean_Squared_Error(self, label, pred):
        return tf.losses.mean_squared_error(labels=label, predictions=pred) 

    def reduce_mean(self, labels, model):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=labels))

    def reduce_mean2(self, labels, label_weight, model):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=labels)
        return tf.reduce_mean(loss * label_weight)