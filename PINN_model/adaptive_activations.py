import tensorflow as tf
import numpy as np
import os

class AdaptiveTanh(tf.keras.layers.Layer):
    def __init__(self, num_neurons, **kwargs):
        super(AdaptiveTanh, self).__init__(**kwargs)
        # Initialize alpha for each neuron in the layer as ones (trainable)
        self.alpha = tf.Variable(initial_value=tf.ones([1, num_neurons]), trainable=True, dtype=tf.float32)

    def call(self, inputs):
        # Apply the adaptive Tanh: tanh(alpha * linear_combination)
        return tf.math.tanh(self.alpha * inputs)


class AdaptiveSwish(tf.keras.layers.Layer):
    def __init__(self, num_neurons, **kwargs):
        super(AdaptiveSwish, self).__init__(**kwargs)
        # Initialize beta for each neuron in the layer as ones (trainable)
        self.beta = tf.Variable(initial_value=tf.ones([1, num_neurons]), trainable=True, dtype=tf.float32)

    def call(self, inputs):
        # Apply the adaptive Swish: x * sigmoid(beta * x)
        return inputs * tf.nn.sigmoid(self.beta * inputs)