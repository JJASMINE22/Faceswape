# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape
from tensorflow.keras.models import Model, Sequential
from custom.CustomLayers import ConvBlock, UpScale

class CreateModel(Model):
    def __init__(self,
                 **kwargs):
        super(CreateModel, self).__init__(**kwargs)

        self.encoder = Sequential([
            ConvBlock(filters=128, kernel_size=(5, 5),
                      strides=(2, 2)),
            ConvBlock(filters=256, kernel_size=(5, 5),
                      strides=(2, 2)),
            ConvBlock(filters=512, kernel_size=(5, 5),
                      strides=(2, 2)),
            ConvBlock(filters=1024, kernel_size=(5, 5),
                      strides=(2, 2)),
            Flatten(),
            Dense(units=1024,
                  kernel_initializer='random_uniform',
                  kernel_regularizer='l2'),
            Dense(units=1024*4*4,
                  kernel_initializer='random_uniform',
                  kernel_regularizer='l2'),
            Reshape(target_shape=(4, 4, 1024)),
            UpScale(filters=512*4, kernel_size=(3, 3),
                    strides=(1, 1))
        ])

        self.former_decoder = Sequential([
            UpScale(filters=256 * 4, kernel_size=(3, 3),
                    strides=(1, 1)),
            UpScale(filters=128 * 4, kernel_size=(3, 3),
                    strides=(1, 1)),
            UpScale(filters=64 * 4, kernel_size=(3, 3),
                    strides=(1, 1)),
            Conv2D(filters=3, kernel_size=(5, 5), padding='same',
                   use_bias=False, kernel_initializer='random_uniform',
                   kernel_regularizer='l2')
        ])

        self.latter_decoder = Sequential([
            UpScale(filters=256 * 4, kernel_size=(3, 3),
                    strides=(1, 1)),
            UpScale(filters=128 * 4, kernel_size=(3, 3),
                    strides=(1, 1)),
            UpScale(filters=64 * 4, kernel_size=(3, 3),
                    strides=(1, 1)),
            Conv2D(filters=3, kernel_size=(5, 5), padding='same',
                   use_bias=False, kernel_initializer='random_uniform',
                   kernel_regularizer='l2')
        ])

    def call(self, inputs, training=None, mask=None, mode='former'):

        feats = self.encoder(inputs)

        if mode == 'former':
            feats = self.former_decoder(feats)
            feats = tf.nn.tanh(feats)
        else:
            feats = self.latter_decoder(feats)
            feats = tf.nn.tanh(feats)

        return feats

