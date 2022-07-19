import tensorflow as tf


def mish(x):
    return x * tf.math.tanh(tf.math.log(1 + tf.exp(x)))
