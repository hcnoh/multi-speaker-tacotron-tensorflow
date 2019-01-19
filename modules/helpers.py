import numpy as np
import tensorflow as tf

from tensorflow.contrib.seq2seq import Helper


class TrainingHelper(Helper):

    def __init__(self, batch_size, reduction_channels, targets, go_frame):
        self._batch_size = batch_size
        self._targets = targets
        self._sequence_length = tf.tile(tf.expand_dims(tf.shape(targets)[1], 0),
                                        [self._batch_size])
        self._reduction_channels = reduction_channels
        self._go_frame = go_frame

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        finished = tf.tile(input=[False], multiples=[self._batch_size])
        next_inputs = tf.tile(
            input=tf.expand_dims(self._go_frame, 0), multiples=[self._batch_size, 1])

        return (finished, next_inputs)

    def sample(self, time, outputs, state, name=None):
        return tf.tile(input=[0], multiples=[self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        next_inputs = self._targets[:, time, :]

        return (finished, next_inputs, state)


class GeneratingHelper(Helper):

    def __init__(self, batch_size, go_frame):
        self._batch_size = batch_size
        self._go_frame = go_frame

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        finished = tf.tile(input=[False], multiples=[self._batch_size])
        next_inputs = tf.tile(
            input=tf.expand_dims(self._go_frame, 0), multiples=[self._batch_size, 1])

        return (finished, next_inputs)

    def sample(self, time, outputs, state, name=None):
        return tf.tile(input=[0], multiples=[self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        next_time = time + 1
        finished = tf.reduce_all(tf.equal(outputs, self._go_frame), axis=1)
        next_inputs = outputs

        return (finished, next_inputs, state)
