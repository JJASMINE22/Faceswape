# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D

class PixelShuffle(Layer):
    def __init__(self,
                 data_format: str=None,
                 **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.data_format = data_format

    def get_config(self):
        config = super(PixelShuffle, self).get_config()

        config.update({
            'data_format': self.data_format
        })

        return config

    def call(self, inputs, *args, **kwargs):

        if self.data_format == 'channels_first':
            batch_size, channels, height, width = inputs.shape
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        else:
            batch_size, height, width, channels = inputs.shape
        feats = tf.reshape(inputs, shape=(batch_size, height, width, 2, 2, -1))
        feats = tf.transpose(feats, perm=[0, 1, 2, 4, 3, 5])
        feats = tf.split(feats, num_or_size_splits=height, axis=1)
        feats = tf.concat([tf.squeeze(feat, axis=1) for feat in feats], axis=2)
        feats = tf.split(feats, num_or_size_splits=width, axis=1)
        feats = tf.concat([tf.squeeze(feat, axis=1) for feat in feats], axis=2)

        if self.data_format == 'channels_first':
            feats = tf.transpose(feats, [0, 3, 1, 2])

        return feats

class ConvBlock(Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str='same',
                 groups: int = 1,
                 data_format: str=None,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        assert not filters % groups
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.data_format = data_format

        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding=padding,
                             groups=groups, data_format=data_format,
                             kernel_initializer='random_uniform',
                             kernel_regularizer='l2')

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "groups": self.groups,
                "data_format": self.data_format
            }
        )

        return config

    def call(self, inputs, *args, **kwargs):

        feats = self.conv2d(inputs)

        feats = tf.nn.leaky_relu(feats, alpha=.1)

        return feats

class UpScale(Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str='same',
                 groups: int = 1,
                 data_format: str = None,
                 **kwargs):
        super(UpScale, self).__init__(**kwargs)
        assert not filters % groups
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.data_format = data_format

        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding=padding,
                             groups=groups, data_format=data_format,
                             kernel_initializer='random_uniform',
                             kernel_regularizer='l2')

        self.pixel_shuffle = PixelShuffle(data_format=data_format)

    def get_config(self):
        config = super(UpScale, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "groups": self.groups,
                "data_format": self.data_format
            }
        )

        return config

    def call(self, inputs, *args, **kwargs):

        feats = self.conv2d(inputs)

        feats = tf.nn.leaky_relu(feats, alpha=.1)

        feats = self.pixel_shuffle(feats)

        return feats
