import sys
from os import listdir, makedirs
from os.path import isfile, join, abspath, dirname
import os
import numpy as np
import math
import json
import cv2
import scipy.misc
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt

from attention_module import *

def cmpooling(fmaps, scale_list, pool_stride):
    # make sure the scale_list is in decending order
    if scale_list[0] - scale_list[1] < 0:
        scale_list = scale_list[::-1]

    # concentric multi-scale pooling
    offset = [0] + [-(scale_list[i + 1] - scale_list[0]) // 2 for i in range(len(scale_list) - 1)]
    pool_maps = []
    for offset, scale in zip(offset, scale_list):
        slice_maps = tf.slice(fmaps, [0, offset, offset, 0],
                              [-1, fmaps.shape[1] - offset * 2, fmaps.shape[2] - offset * 2, -1])
        pool_map = tf.nn.max_pool2d(slice_maps, scale, pool_stride, "VALID")
        pool_maps.append(pool_map)

    # assert same shape for all pool_map
    for i in range(len(pool_maps) - 1):
        assert pool_maps[i].shape[1:] == pool_maps[-1].shape[1:]
    return pool_maps


# Concat the feature maps in different scale and convolution once. (paper version)
class Monocular(tf.keras.layers.Layer):
    def __init__(self, filters, ksize, **kwargs):
        super(Monocular, self).__init__(**kwargs)
        self.filters = filters
        self.ksize = ksize

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(self.filters, self.ksize, input_shape=input_shape, activation='relu',
                                           padding='same', trainable=True, )

    def call(self, fmaps, scale_list, pool_stride):
        pool_maps = cmpooling(fmaps, scale_list, pool_stride)
        pool_maps = tf.concat(pool_maps, axis=-1)
        return self.conv(pool_maps)

    def get_config(self):
        config = {"filters": self.filters, "ksize": self.ksize}
        # config = {"ksize": self.ksize}
        base_config = super(Monocular, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class S3DModel():
    def __init__(self, weights=''):
        self.build_s3d_model()
        if weights:
            self.model.load_weights(weights)

    def build_s3d_model(self):
        input_shape = (224, 224, 3)
        scale_list = [1, 3, 5]
        eps = 1e-7
        test = True
        img_input_left = layers.Input(shape=input_shape, name='Input_left')
        img_input_right = layers.Input(shape=input_shape, name='Input_right')
        # print('img_input_left:',img_input_left)

        # parallax augmentation
        parallax = img_input_left - img_input_right
        left = tf.keras.layers.Concatenate(axis=-1)([img_input_left, -parallax])
        right = tf.keras.layers.Concatenate(axis=-1)([img_input_right, parallax])
        print('left:',left.shape)
        print('right:',right.shape)
        # Block 1
        # (B,110,110,64)
        left_1_1 = Monocular(64, 3, input_shape=input_shape, name='mono1_left_1_1')(left, scale_list=scale_list,
                                                                                    pool_stride=2)
        # print('left_1_1:',left_1_1)
        # exit()
        left_1_2 = Monocular(64, 3, name='mono1_left_1_2')(left_1_1, scale_list=scale_list,
                                                           pool_stride=2)
        # print('left_1_2:',left_1_2)
        # exit()
        right_1_1 = Monocular(64, 3, input_shape=input_shape, name='mono1_right_1_1')(right, scale_list=scale_list,
                                                                                      pool_stride=2)
        # print('right_1_1:',right_1_1)
        # exit()
        right_1_2 = Monocular(64, 3, name='mono1_right_1_2')(right_1_1, scale_list=scale_list,
                                                             pool_stride=2)
        # print('right_1_2:',right_1_2)
        # exit()
        # Block 2
        # SPADE #
        right_2_fea = layers.Conv2D(128, (3, 3),
                                  activation='relu',
                                  padding='same',
                                  trainable=True,
                                  name='right_2_fea')(right_1_2)
        left_2_sp1 = layers.Conv2D(128, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='left_2_sp1')(left_1_2)

        left_2_gamma = layers.Conv2D(128, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='left_2_gamma')(right_2_fea)
        left_2_beta = layers.Conv2D(128, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='left_2_beta')(right_2_fea)

        left_2_input = left_2_sp1 * left_2_gamma + left_2_beta

        left_2_1 = Monocular(128, 3, name='mono2_left_2_1')(left_2_input,
                                                            scale_list=scale_list, pool_stride=1)

        left_2_2 = Monocular(128, 3, name='mono2_left_2_2')(left_2_1, scale_list=scale_list,
                                                            pool_stride=1)
        # print('left_2_1:',left_2_1)
        # print('left_2_2:',left_2_2)
        # exit()
        # SPADE #
        left_2_fea = layers.Conv2D(128, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   trainable=True,
                                   name='left_2_fea')(left_1_2)
        right_2_sp1 = layers.Conv2D(128, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='right_2_sp1')(right_1_2)
        right_2_gamma = layers.Conv2D(128, (3, 3),
                                      activation=None,
                                      padding='same',
                                      trainable=True,
                                      name='right_2_gamma')(left_2_fea)
        right_2_beta = layers.Conv2D(128, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='right_2_beta')(left_2_fea)
        right_2_input = right_2_sp1 * right_2_gamma + right_2_beta

        right_2_1 = Monocular(128, 3, name='mono2_right_2_1')(right_2_input,
                                                              scale_list=scale_list,
                                                              pool_stride=1)
        right_2_2 = Monocular(128, 3, name='mono2_right_2_2')(right_2_1, scale_list=scale_list,
                                                              pool_stride=1)

        # Block 3
        # SPADE #
        right_3_fea = layers.Conv2D(256, (3, 3),
                                  activation='relu',
                                  padding='same',
                                  trainable=True,
                                  name='right_3_fea')(right_2_2)
        left_3_sp1 = layers.Conv2D(256, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='left_3_sp1')(left_2_2)
        left_3_gamma = layers.Conv2D(256, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='left_3_gamma')(right_3_fea)
        left_3_beta = layers.Conv2D(256, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='left_3_beta')(right_3_fea)

        left_3_input = left_3_sp1 * left_3_gamma + left_3_beta

        left_3_1 = Monocular(256, 3, name='mono3_left_3_1')(left_3_input,
                                                            scale_list=scale_list,
                                                            pool_stride=1)
        left_3_2 = Monocular(256, 3, name='mono3_left_3_2')(left_3_1, scale_list=scale_list,
                                                            pool_stride=1)
        left_3_3 = Monocular(256, 3, name='mono3_left_3_3')(left_3_2, scale_list=scale_list,
                                                            pool_stride=1)
        # print('left_3_1:',left_3_1)
        # print('left_3_2:',left_3_2)
        # print('left_3_3:',left_3_3)
        # exit()
        # SPADE #
        left_3_fea = layers.Conv2D(256, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   trainable=True,
                                   name='left_3_fea')(left_2_2)
        right_3_sp1 = layers.Conv2D(256, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='right_3_sp1')(right_2_2)
        right_3_gamma = layers.Conv2D(256, (3, 3),
                                      activation=None,
                                      padding='same',
                                      trainable=True,
                                      name='right_3_gamma')(left_3_fea)
        right_3_beta = layers.Conv2D(256, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='right_3_beta')(left_3_fea)
        right_3_input = right_3_sp1 * right_3_gamma + right_3_beta
        right_3_1 = Monocular(256, 3, name='mono3_right_3_1')(right_3_input,
                                                              scale_list=scale_list,
                                                              pool_stride=1)
        right_3_2 = Monocular(256, 3, name='mono3_right_3_2')(right_3_1, scale_list=scale_list,
                                                              pool_stride=1)
        right_3_3 = Monocular(256, 3, name='mono3_right_3_3')(right_3_2, scale_list=scale_list,
                                                              pool_stride=1)

        # Block 4
        # SPADE #
        right_4_fea = layers.Conv2D(512, (3, 3),
                                  activation='relu',
                                  padding='same',
                                  trainable=True,
                                  name='right_4_fea')(right_3_3)
        left_4_sp1 = layers.Conv2D(512, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='left_4_sp1')(left_3_3)
        left_4_gamma = layers.Conv2D(512, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='left_4_gamma')(right_4_fea)
        left_4_beta = layers.Conv2D(512, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='left_4_beta')(right_4_fea)

        left_4_input = left_4_sp1 * left_4_gamma + left_4_beta
        left_4_1 = Monocular(512, 3, name='mono4_left_4_1')(left_4_input,
                                                            scale_list=scale_list,
                                                            pool_stride=1)
        left_4_2 = Monocular(512, 3, name='mono4_left_4_2')(left_4_1, scale_list=scale_list,
                                                            pool_stride=1)
        left_4_3 = Monocular(512, 3, name='mono4_left_4_3')(left_4_2, scale_list=scale_list,
                                                            pool_stride=1)

        # print('left_4_1:',left_4_1)
        # print('left_4_2:',left_4_2)
        # print('left_4_3:',left_4_3)
        # exit()
        # SPADE #
        left_4_fea = layers.Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   trainable=True,
                                   name='left_4_fea')(left_3_3)
        right_4_sp1 = layers.Conv2D(512, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='right_4_sp1')(right_3_3)
        right_4_gamma = layers.Conv2D(512, (3, 3),
                                      activation=None,
                                      padding='same',
                                      trainable=True,
                                      name='right_4_gamma')(left_4_fea)
        right_4_beta = layers.Conv2D(512, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='right_4_beta')(left_4_fea)
        right_4_input = right_4_sp1 * right_4_gamma + right_4_beta
        right_4_1 = Monocular(512, 3, name='mono4_right_4_1')(right_4_input,
                                                              scale_list=scale_list,
                                                              pool_stride=1)
        right_4_2 = Monocular(512, 3, name='mono4_right_4_2')(right_4_1, scale_list=scale_list,
                                                              pool_stride=1)
        right_4_3 = Monocular(512, 3, name='mono4_right_4_3')(right_4_2, scale_list=scale_list,
                                                              pool_stride=1)

        # Block 5
        # SPADE #
        right_5_fea = layers.Conv2D(512, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='right_5_fea')(right_4_3)
        left_5_sp1 = layers.Conv2D(512, (3, 3),
                                   activation=None,
                                   padding='same',
                                   trainable=True,
                                   name='left_5_sp1')(left_4_3)
        left_5_gamma = layers.Conv2D(512, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='left_5_gamma')(right_5_fea)
        left_5_beta = layers.Conv2D(512, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='left_5_beta')(right_5_fea)

        left_5_input = left_5_sp1 * left_5_gamma + left_5_beta


        left_5_1 = Monocular(512, 3, name='mono5_left_5_1')(left_5_input,
                                                            scale_list=scale_list,
                                                            pool_stride=1)
        left_5_2 = Monocular(512, 3, name='mono5_left_5_2')(left_5_1, scale_list=scale_list,
                                                            pool_stride=1)
        left_5_3 = Monocular(512, 3, name='mono5_left_5_3')(left_5_2, scale_list=scale_list,
                                                            pool_stride=1)

        # print('left_5_1:', left_5_1)
        # print('left_5_2:',left_5_2)
        # print('left_5_3:',left_5_3)
        # exit()
        # SPADE # *********
        left_5_fea = layers.Conv2D(512, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   trainable=True,
                                   name='left_5_fea')(left_4_3)
        right_5_sp1 = layers.Conv2D(512, (3, 3),
                                    activation=None,
                                    padding='same',
                                    trainable=True,
                                    name='right_5_sp1')(right_4_3)
        right_5_gamma = layers.Conv2D(512, (3, 3),
                                      activation=None,
                                      padding='same',
                                      trainable=True,
                                      name='right_5_gamma')(left_5_fea)
        right_5_beta = layers.Conv2D(512, (3, 3),
                                     activation=None,
                                     padding='same',
                                     trainable=True,
                                     name='right_5_beta')(left_5_fea)
        right_5_input = right_5_sp1 * right_5_gamma + right_5_beta
        right_5_1 = Monocular(512, 3, name='mono5_right_5_1')(right_5_input,
                                                              scale_list=scale_list, pool_stride=1)

        right_5_2 = Monocular(512, 3, name='mono5_right_5_2')(right_5_1, scale_list=scale_list,
                                                              pool_stride=1)
        right_5_3 = Monocular(512, 3, name='mono5_right_5_3')(right_5_2, scale_list=scale_list,
                                                              pool_stride=1)
        # Left Pyramid feature representation
        left_5_3_up = tf.image.resize(left_5_3, (left_4_3.shape[1], left_4_3.shape[2]),
                                      method=tf.image.ResizeMethod.BILINEAR)

        left_5_3_cat = layers.concatenate([left_5_3_up, left_4_3], axis=-1)

        left_py_1_1 = layers.Conv2D(512, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='left_py_1_1')(left_5_3_cat)
        left_py_1_2 = layers.Conv2D(128, (1, 1),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='left_py_1_2')(left_py_1_1)
        #attention
        left_1_a1 = left_5_3_cat
        left_1_a1 = left_1_a1 * 0.1
        left_1_avg_pool = tf.reduce_mean(left_1_a1, axis=[3], keepdims=True)
        left_1_max_pool = tf.reduce_max(left_1_a1, axis=[3], keepdims=True)
        left_1_concat = tf.concat([left_1_avg_pool, left_1_max_pool], 3)

        left_1_concat1 = layers.Conv2D(1,
                                  kernel_size=[3, 3],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  use_bias=False,
                                  name='conv_left_1_concat1')(left_1_concat)

        left_1_concat2 = tf.sigmoid(left_1_concat1, 'sigmoid')
        left_1_concat3 = left_1_a1 * left_1_concat2

        left_1_a2 = left_5_3_cat + left_1_concat3
        left_1_a3 = layers.BatchNormalization()(left_1_a2,training=True)
        left_1_a4 = tf.nn.relu(left_1_a3)
        left_1_a5 = layers.Conv2D(128, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         trainable=True,
                                         name='left_1_a5')(left_1_a4)
        left_1_a = tf.add(left_1_a5, left_py_1_2)


        #####the second global unit
        left_4_3_up = tf.image.resize(left_1_a, (left_3_3.shape[1], left_3_3.shape[2]),
                                      method=tf.image.ResizeMethod.BILINEAR)
        left_4_3_cat = layers.concatenate([left_4_3_up, left_3_3], axis=-1)
        left_py_2_1 = layers.Conv2D(256, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='left_py_2_1')(left_4_3_cat)

        left_py_2_2 = layers.Conv2D(128, (1, 1),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='left_py_2_2')(left_py_2_1)
        # attention
        left_2_a1 = left_4_3_cat
        left_2_a1 = left_2_a1 * 0.1
        left_2_avg_pool = tf.reduce_mean(left_2_a1, axis=[3], keepdims=True)
        left_2_max_pool = tf.reduce_max(left_2_a1, axis=[3], keepdims=True)
        left_2_concat = tf.concat([left_2_avg_pool, left_2_max_pool], 3)

        left_2_concat1 = layers.Conv2D(1, kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="same",
                                          activation=None,
                                          use_bias=False,
                                          name='conv_left_2_concat1')(left_2_concat)

        left_2_concat2 = tf.sigmoid(left_2_concat1, 'sigmoid')
        left_2_concat3 = left_2_a1 * left_2_concat2

        left_2_a2 = left_4_3_cat + left_2_concat3
        left_2_a3 = layers.BatchNormalization()(left_2_a2,training=True)
        left_2_a4 = tf.nn.relu(left_2_a3)
        left_2_a5 = layers.Conv2D(128, (1, 1),
                                  activation='relu',
                                  padding='same',
                                  trainable=True,
                                  name='left_2_a5')(left_2_a4)
        left_2_a = tf.add(left_2_a5, left_py_2_2)

        ##########
        left_3_3_up = tf.image.resize(left_2_a, (left_2_2.shape[1], left_2_2.shape[2]),
                                      method=tf.image.ResizeMethod.BILINEAR)
        left_3_3_cat = layers.concatenate([left_3_3_up, left_2_2], axis=-1)
        left_py_3_1 = layers.Conv2D(128, (1, 1),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='left_py_3_1')(left_3_3_cat)

        # attention
        left_3_a1 = left_3_3_cat
        left_3_a1 = left_3_a1 * 0.1
        left_3_avg_pool = tf.reduce_mean(left_3_a1, axis=[3], keepdims=True)
        left_3_max_pool = tf.reduce_max(left_3_a1, axis=[3], keepdims=True)
        left_3_concat = tf.concat([left_3_avg_pool, left_3_max_pool], 3)

        left_3_concat1 = layers.Conv2D(1, kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="same",
                                          activation=None,
                                          use_bias=False,
                                          name='conv_left_3_concat1')(left_3_concat)

        left_3_concat2 = tf.sigmoid(left_3_concat1, 'sigmoid')
        left_3_concat3 = left_3_a1 * left_3_concat2

        left_3_a2 = left_3_3_cat + left_3_concat3
        left_3_a3 = layers.BatchNormalization()(left_3_a2,training=True)
        left_3_a4 = tf.nn.relu(left_3_a3)
        left_3_a5 = layers.Conv2D(128, (1, 1),
                                  activation='relu',
                                  padding='same',
                                  trainable=True,
                                  name='left_3_a5')(left_3_a4)
        left_3_a = tf.add(left_3_a5, left_py_3_1)

        # right Pyramid feature representation
        right_5_3_up = tf.image.resize(right_5_3, (right_4_3.shape[1], right_4_3.shape[2]),
                                       method=tf.image.ResizeMethod.BILINEAR)

        right_5_3_cat = layers.concatenate([right_5_3_up, right_4_3], axis=-1)

        right_py_1_1 = layers.Conv2D(512, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     trainable=True,
                                     name='right_py_1_1')(right_5_3_cat)
        right_py_1_2 = layers.Conv2D(128, (1, 1),
                                     activation='relu',
                                     padding='same',
                                     trainable=True,
                                     name='right_py_1_2')(right_py_1_1)
        # attention
        right_1_a1 = right_5_3_cat
        right_1_a1 = right_1_a1 * 0.1
        right_1_avg_pool = tf.reduce_mean(right_1_a1, axis=[3], keepdims=True)
        right_1_max_pool = tf.reduce_max(right_1_a1, axis=[3], keepdims=True)
        right_1_concat = tf.concat([right_1_avg_pool, right_1_max_pool], 3)

        right_1_concat1 = layers.Conv2D(1,kernel_size=[3, 3],
                                           strides=[1, 1],
                                           padding="same",
                                           activation=None,
                                           use_bias=False,
                                           name='conv_right_1_concat1')(right_1_concat)

        right_1_concat2 = tf.sigmoid(right_1_concat1, 'sigmoid')
        right_1_concat3 = right_1_a1 * right_1_concat2

        right_1_a2 = right_5_3_cat + right_1_concat3
        right_1_a3 = layers.BatchNormalization()(right_1_a2,training=True)
        right_1_a4 = tf.nn.relu(right_1_a3)
        right_1_a5 = layers.Conv2D(128, (1, 1),
                                   activation='relu',
                                   padding='same',
                                   trainable=True,
                                   name='right_1_a5')(right_1_a4)
        right_1_a = tf.add(right_1_a5, right_py_1_2)

        #####the second global unit
        right_4_3_up = tf.image.resize(right_1_a, (right_3_3.shape[1], right_3_3.shape[2]),
                                       method=tf.image.ResizeMethod.BILINEAR)
        right_4_3_cat = layers.concatenate([right_4_3_up, right_3_3], axis=-1)
        right_py_2_1 = layers.Conv2D(256, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     trainable=True,
                                     name='right_py_2_1')(right_4_3_cat)

        right_py_2_2 = layers.Conv2D(128, (1, 1),
                                     activation='relu',
                                     padding='same',
                                     trainable=True,
                                     name='right_py_2_2')(right_py_2_1)
        # attention
        right_2_a1 = right_4_3_cat
        right_2_a1 = right_2_a1 * 0.1
        right_2_avg_pool = tf.reduce_mean(right_2_a1, axis=[3], keepdims=True)
        right_2_max_pool = tf.reduce_max(right_2_a1, axis=[3], keepdims=True)
        right_2_concat = tf.concat([right_2_avg_pool, right_2_max_pool], 3)

        right_2_concat1 = layers.Conv2D(1,kernel_size=[3, 3],
                                           strides=[1, 1],
                                           padding="same",
                                           activation=None,
                                           use_bias=False,
                                           name='conv_right_2_concat1')(right_2_concat)

        right_2_concat2 = tf.sigmoid(right_2_concat1, 'sigmoid')
        right_2_concat3 = right_2_a1 * right_2_concat2

        right_2_a2 = right_4_3_cat + right_2_concat3
        right_2_a3 = layers.BatchNormalization()(right_2_a2,training=True)
        right_2_a4 = tf.nn.relu(right_2_a3)
        right_2_a5 = layers.Conv2D(128, (1, 1),
                                   activation='relu',
                                   padding='same',
                                   trainable=True,
                                   name='right_2_a5')(right_2_a4)
        right_2_a = tf.add(right_2_a5, right_py_2_2)

        ##########
        right_3_3_up = tf.image.resize(right_2_a, (right_2_2.shape[1], right_2_2.shape[2]),
                                       method=tf.image.ResizeMethod.BILINEAR)
        right_3_3_cat = layers.concatenate([right_3_3_up, right_2_2], axis=-1)
        right_py_3_1 = layers.Conv2D(128, (1, 1),
                                     activation='relu',
                                     padding='same',
                                     trainable=True,
                                     name='right_py_3_1')(right_3_3_cat)

        # attention
        right_3_a1 = right_3_3_cat
        right_3_a1 = right_3_a1 * 0.1
        right_3_avg_pool = tf.reduce_mean(right_3_a1, axis=[3], keepdims=True)
        right_3_max_pool = tf.reduce_max(right_3_a1, axis=[3], keepdims=True)
        right_3_concat = tf.concat([right_3_avg_pool, right_3_max_pool], 3)

        right_3_concat1 = layers.Conv2D(1,kernel_size=[3, 3],
                                           strides=[1, 1],
                                           padding="same",
                                           activation=None,
                                           use_bias=False,
                                           name='conv_right_3_concat1')(right_3_concat)

        right_3_concat2 = tf.sigmoid(right_3_concat1, 'sigmoid')
        right_3_concat3 = right_3_a1 * right_3_concat2

        right_3_a2 = right_3_3_cat + right_3_concat3
        right_3_a3 = layers.BatchNormalization()(right_3_a2,training=True)
        right_3_a4 = tf.nn.relu(right_3_a3)
        right_3_a5 = layers.Conv2D(128, (1, 1),
                                   activation='relu',
                                   padding='same',
                                   trainable=True,
                                   name='right_3_a5')(right_3_a4)
        right_3_a = tf.add(right_3_a5, right_py_3_1)

        # print('right_py_3_1:',right_py_3_1.shape)
        # exit()
        # feature map fusion
        left_6_1_up = tf.image.resize(left_5_3, (left_3_a.shape[1], left_3_a.shape[2]),
                                      method=tf.image.ResizeMethod.BILINEAR)
        left_6_2_up = layers.Conv2D(256, (3, 3),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='left_6_2_up')(left_6_1_up)
        left_6_3_up = layers.Conv2D(128, (1, 1),
                                    activation='relu',
                                    padding='same',
                                    trainable=True,
                                    name='left_6_3_up')(left_6_2_up)
        right_6_1_up = tf.image.resize(right_5_3, (right_3_a.shape[1], right_3_a.shape[2]),
                                       method=tf.image.ResizeMethod.BILINEAR)
        right_6_2_up = layers.Conv2D(256, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     trainable=True,
                                     name='right_6_2_up')(right_6_1_up)
        right_6_3_up = layers.Conv2D(128, (1, 1),
                                     activation='relu',
                                     padding='same',
                                     trainable=True,
                                     name='right_6_3_up')(right_6_2_up)

        concat_layer = layers.concatenate([left_6_3_up, right_6_3_up, left_3_a, right_3_a], axis=-1)
        concat_layer_1 = layers.Conv2D(256, (3, 3),
                                       activation='relu',
                                       padding='same',
                                       trainable=True,
                                       name='concat_layer_1')(concat_layer)
        concat_layer_2 = layers.Conv2D(128, (1, 1),
                                       activation='relu',
                                       padding='same',
                                       trainable=True,
                                       name='concat_layer_2')(concat_layer_1)
        sal_map_layer = layers.Conv2D(1, (1, 1),
                                      name='saliency_map',
                                      trainable=True,
                                      activation='sigmoid',
                                      kernel_initializer=keras.initializers.Zeros(),
                                      bias_initializer=keras.initializers.Zeros())(concat_layer_2)
        #print('sal_map_layer:',sal_map_layer)
        if test:
            min_per_image = tf.reduce_min(sal_map_layer, axis=(1, 2, 3), keep_dims=True)
            sal_map_layer -= min_per_image

            max_per_image = tf.reduce_max(sal_map_layer, axis=(1, 2, 3), keep_dims=True)
            sal_map_layer = tf.divide(sal_map_layer, eps + max_per_image, name="output")
        
        

        self.model = Model(inputs=[img_input_left, img_input_right], outputs=sal_map_layer)
        self.model.summary()

    def compute_saliency(self, img_path=None, img=None):
        vgg_mean = np.array([123, 116, 103])

        if img_path:
            img_fine = img_to_array(load_img(img_path,
                                             grayscale=False,
                                             target_size=(224, 224),
                                             interpolation='nearest'))
 
            img_right_path = img_path.replace("left", "right");

            print('img_path:', img_path)
            print('img_right_path:', img_right_path)


            img_coarse = img_to_array(load_img(img_right_path,
                                               grayscale=False,
                                               target_size=(224, 224),
                                               interpolation='nearest'))

        else:
            img_fine = img.copy().resize((224, 224))
            img_coarse = img.copy().resize((224, 224))

        img_fine -= vgg_mean
        img_coarse -= vgg_mean

        img_fine = img_fine[None, :] / 255
        img_coarse = img_coarse[None, :] / 255

        smap = np.squeeze(self.model.predict([img_fine, img_coarse], batch_size=1, verbose=0))

        if img_path:
            img = cv2.imread(img_path)
          
            h, w = img.shape[:2]
        else:
            w, h = img.size[:2]

        smap = cv2.resize(smap, (w, h), interpolation=cv2.INTER_CUBIC)
        smap = cv2.GaussianBlur(smap, (75, 75), 25, cv2.BORDER_DEFAULT)

        return smap
    


if __name__ == "__main__":
    main(sys.argv)
