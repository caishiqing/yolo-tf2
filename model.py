from tensorflow.keras import layers
import tensorflow as tf
from utils import *


def CBX(x, filters, kernel_size, strides=1, activation=None):
    if strides == 1:
        padding = "same"

    x = layers.Conve2D(filters, kernel_size,
                       strides=strides,
                       padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def CBL(x, filters, kernel_size, strides=1):
    return CBX(x, filters, kernel_size, strides,
               activation=layers.LeakyReLU())


def CBM(x, filters, kernel_size, strides=1):
    return CBX(x, filters, kernel_size, strides,
               activation=mish)


def res(x, filters, cbx=CBL):
    prev = x
    x = cbx(x, filters//2, 1)
    x = cbx(x, filters, 3)
    return layers.Add()([prev, x])


def resX(x, filters, strides, num=1, cbx=CBL):
    x = cbx(x, filters, 3, strides)
    for _ in range(num):
        x = res(x, filters, cbx)
    return x


def CSPX(x, filters, num=1, cbx=CBM):
    # down sampling
    x = cbx(x, filters, 3, 2)
    prev = x
    x = resX(x, filters, 1, num=num, cbx=cbx)
    a = cbx(x, filters, 3, 1)
    b = cbx(prev, filters, 3, 1)
    x = layers.Concatenate()([a, b])
    x = cbx(x, filters, 3, 1)
    return x


def SPP(x, sizes):
    xs = [x]
    for size in sizes:
        xs.append(layers.MaxPool2D(size, padding="same"))
    return layers.Concatenate()(xs)
