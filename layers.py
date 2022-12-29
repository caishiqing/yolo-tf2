from tensorflow.keras import layers
import tensorflow as tf
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


def CSP1_X(x, filters, num=1, activation="leaky"):
    pre = x
    x = conv(x, filters, 3, 1, activation)
    for _ in range(num):
        x = res(x, filters, activation)

    a = layers.Conv2D(filters, 3, padding="same")(x)
    b = layers.Conv2D(filters, 3, padding="same")(pre)
    x = layers.Concatenate()([a, b])
    x = layers.BatchNormalization()(x)
    x = act(activation)(x)
    x = conv(x, filters, 3, 1, activation)
    return x


def CSP2_X(x, filters, num=1, activation="leaky"):
    pre = x
    x = conv(x, filters, 3, 1, activation)
    for _ in range(2 * num):
        x = conv(x, filters, 3, 1, activation)

    a = layers.Conv2D(filters, 3, padding="same")(x)
    b = layers.Conv2D(filters, 3, padding="same")(pre)
    x = layers.Concatenate()([a, b])
    x = layers.BatchNormalization()(x)
    x = act(activation)(x)
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


def Focus(x, filters=32, activation="leaky"):
    x1 = layers.Lambda(lambda x: x[:, ::2, ::2, :], name='stride1')(x)
    x2 = layers.Lambda(lambda x: x[:, 1::2, ::2, :], name='stride2')(x)
    x3 = layers.Lambda(lambda x: x[:, ::2, 1::2, :], name='stride3')(x)
    x4 = layers.Lambda(lambda x: x[:, 1::2, 1::2, :], name='stride4')(x)
    x = layers.Concatenate(-1)([x1, x2, x3, x4])
    x = conv(x, filters, 3, activation=activation)
    return x


@tf.keras.utils.register_keras_serializable("yloo-tf2")
class Head(layers.Layer):
    def __init__(self, anchor_prior_params, **kwargs):
        super(Head, self).__init__(**kwargs)
        self.anchor_prior_params = anchor_prior_params
        self.h_normed, self.w_normed = anchor_prior_params

    def build(self, input_shape):
        _, fmap_h, fmap_w, dim = input_shape
        self.nc = dim - 5
        self.fmap_h = fmap_h
        self.fmap_w = fmap_w
        self.cy = tf.range(0, self.fmap_h, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
        self.cx = tf.range(0, self.fmap_w, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
        self.built = True

    def call(self, logits):
        pos_logits = logits[..., :4]
        prb_logits = logits[..., 4]
        cls_logits = logits[..., 5:]
        pos = tf.nn.sigmoid(pos_logits)
        prb = tf.nn.sigmoid(prb_logits)
        cls = tf.nn.softmax(cls_logits, axis=-1)

        # all positionn infomations are normalized to [0, 1]
        by = (pos[..., 0] + self.cy) / tf.cast(self.fmap_h, tf.float32)
        bx = (pos[..., 1] + self.cx) / tf.cast(self.fmap_w, tf.float32)
        bh = pos[..., 2] * self.h_normed
        bw = pos[..., 3] * self.w_normed
        y1 = by - bh / 2
        x1 = bx - bw / 2
        y2 = by + bh / 2
        x2 = bx + bw / 2
        y1 = tf.where(tf.less(y1, 0), 0.0, y1)
        x1 = tf.where(tf.less(x1, 0), 0.0, x1)
        y2 = tf.where(tf.greater(y2, 1), 1.0, y2)
        x2 = tf.where(tf.greater(x2, 1), 1.0, x2)

        pos = tf.stack([y1, x1, y2, x2], axis=-1)
        pos = tf.reshape(pos, [-1, self.fmap_w * self.fmap_h, 4])
        prb = tf.reshape(prb, [-1, self.fmap_w * self.fmap_h])
        cls = tf.reshape(cls, [-1, self.fmap_w * self.fmap_h, self.nc])
        return pos, prb, cls

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        pos_shape = (b, h, w, 4)
        prb_shape = (b, h, w)
        cls_shape = (b, h, w, c-5)
        return pos_shape, prb_shape, cls_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "anchor_prior_params": self.anchor_prior_params
            }
        )
        return config


def Heads(inputs, anchor_priors):
    assert len(inputs) == len(anchor_priors)
    all_pos, all_prb, all_cls = [], [], []

    for logits, priors in zip(inputs, anchor_priors):
        # n_scale loops
        num_anchors = len(priors)
        anchor_logits = layers.Lambda(
            lambda x: tf.split(x, num_anchors, axis=-1),
            output_shape=(
                tf.shape(logits)[1],
                tf.shape(logits)[2],
                tf.shape(logits)[3] // num_anchors
            )
        )(logits)
        for x, anchor_prior_params in zip(anchor_logits, priors):
            # n_anchor loops
            pos, prb, cls = Head(anchor_prior_params)(x)
            all_pos.append(pos)
            all_prb.append(prb)
            all_cls.append(cls)

    # (batch, logits_w * logits_h * n_anchor * n_scale, 4 or None or n_class)
    pos = layers.Concatenate(1, name="position")(all_pos)
    prb = layers.Concatenate(1, name="confidence")(all_prb)
    cls = layers.Concatenate(1, name="classify")(all_cls)
    return pos, prb, cls
