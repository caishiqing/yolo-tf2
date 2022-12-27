from tensorflow.keras import layers
import tensorflow as tf
from torch import xlogy
from utils import *


def act(name="leaky"):
    activation = layers.LeakyReLU() if name == "leaky" else mish
    return layers.Activation(activation)


def conv(x, filters, kernel_size, strides=1, activation="leaky"):
    if strides == 1:
        padding = "same"
    else:
        x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = "valid"

    x = layers.Conv2D(filters, kernel_size,
                      strides=strides,
                      padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = act(activation)(x)
    return x


def convX(x, fs, ks, strides=None, activation="leaky"):
    assert len(fs) == len(ks)
    if strides is None:
        strides = [1]*len(fs)

    for f, k, s in zip(fs, ks, strides):
        x = conv(x, f, k, s, activation)

    return x


def res(x, filters, activation="leaky"):
    prev = x
    x = conv(x, filters//2, 1, 1, activation)
    x = conv(x, filters, 3, 1, activation)
    return layers.Add()([prev, x])


def resX(x, filters, num=1, activation="leaky"):
    x = conv(x, filters, 3, 2, activation)
    for _ in range(num):
        x = res(x, filters, activation)
    return x


def CSPX(x, filters, num=1, activation="mish"):
    # down sampling
    x = conv(x, filters, 3, 2, activation)
    prev = x
    x = conv(x, filters, 1, 1, activation)
    for _ in range(num):
        x = res(x, filters, activation)

    a = conv(x, filters, 3, 1, activation)
    b = conv(prev, filters, 3, 1, activation)
    x = layers.Concatenate()([a, b])
    x = conv(x, filters, 3, 1, activation)
    return x


def SPP(x, sizes):
    xs = []
    for size in sizes:
        xs.append(layers.MaxPool2D(size, padding="same", strides=1)(x))
    return layers.Concatenate()(xs)


def FPN(x, x_pre, filters, activation="leaky"):
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = act(activation)(x)
    x = layers.Concatenate()([x, x_pre])
    return x


def PAN(x, x_pre, filters, activation="leaky"):
    x = conv(x, filters, 3, 2, activation)
    x = layers.Concatenate()([x, x_pre])
    return x


class Head(layers.Layer):
    def __init__(self, image_size, anchor_prior_params, **kwargs):
        super(Head, self).__init__(**kwargs)
        self.img_size = image_size
        self.anchor_prior_params = anchor_prior_params
        self.width, self.height = image_size
        self.w_normed, self.h_normed = anchor_prior_params

    def call(self, logits):
        pos_logits = logits[..., :4]
        prb_logits = logits[..., 4]
        cls_logits = logits[..., 5:]
        pos = tf.nn.sigmoid(pos_logits)
        prb = tf.nn.sigmoid(prb_logits)
        cls = tf.nn.softmax(cls_logits, axis=-1)

        # all positionn infomations are normalized to [0, 1]
        logits_w = tf.shape(pos)[1]
        logits_h = tf.shape(pos)[2]
        cx = tf.range(0, logits_w, dtype=pos.dtype)[tf.newaxis, :, tf.newaxis]
        cy = tf.range(0, logits_h, dtype=pos.dtype)[tf.newaxis, tf.newaxis, :]
        bx = (pos[..., 0] + cx) / tf.cast(logits_w, cx.dtype)
        by = (pos[..., 1] + cy) / tf.cast(logits_h, cx.dtype)
        bw = pos[..., 2] * self.w_normed
        bh = pos[..., 3] * self.h_normed

        # rescale to origin image size
        bx *= self.width
        by *= self.height
        bw *= self.width
        bh *= self.height

        pos = tf.stack([bx, by, bw, bh], axis=-1)
        pos = tf.reshape(pos, [-1, logits_w * logits_h, 4])
        prb = tf.reshape(prb, [-1, logits_w * logits_h, 1])
        cls = tf.reshape(cls, [-1, logits_w * logits_h, tf.shape(cls)[-1]])
        y = tf.concat([pos, prb, cls], axis=-1)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "anchor_prior_params": self.anchor_prior_params
            }
        )
        return config


def Heads(inputs, image_size, anchor_priors):
    assert len(inputs) == len(anchor_priors)
    outputs = []

    for logits, priors in zip(inputs, anchor_priors):
        num_anchors = len(priors)
        anchor_logits = layers.Lambda(
            lambda x: tf.split(x, num_anchors, axis=-1),
            output_shape=(
                tf.shape(logits)[1],
                tf.shape(logits)[2],
                tf.shape(logits) // num_anchors
            )
        )(logits)
        for x, anchor_prior_params in zip(anchor_logits, priors):
            # anchor_prior_params = (w_normed, h_normed)
            y = Head(image_size, anchor_prior_params)(x)
            outputs.append(y)

    # (batch, logits_w * logits_h * n_anchor * n_scale, 4 + 1 + num_classes)
    y = layers.Concatenate(1)(outputs)
    return y
