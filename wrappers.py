import sys, re
from pprint import pprint

import tensorflow as tf
import tensorflow.python.platform

from config import baseConfig

class Pooling2D:
    def __init__(self, name, pool_size=(3, 3), strides=(2, 2), padding='VALID'):
        self.name    = name
        self.ksize   = [1, pool_size[0], pool_size[1], 1]
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding
    
class MaxPooling2D(Pooling2D):
    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        with tf.name_scope(self.name) as scope:
            return tf.nn.max_pool(h, ksize=self.ksize, strides=self.strides, padding=self.padding)

class AveragePooling2D(Pooling2D):
    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        with tf.name_scope(self.name) as scope:
            return tf.nn.avg_pool(h, ksize=self.ksize, strides=self.strides, padding=self.padding)

class GlobalAveragePooling2D:
    def __init__(self, name):
        self.name = name

    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        if baseConfig.dataFormat == 'channels_last':
            return tf.reduce_mean(h, reduction_indices=[1, 2])
        else:
            return tf.reduce_mean(h, reduction_indices=[2, 3])

class Conv2D:
    def __init__(self, name, channel,  filter_size=(3, 3), strides=(1, 1), padding='VALID'):
        self.name    = name
        self.channel = channel
        self.filter_size   = filter_size
        self.strides = strides
        self.padding = padding

    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        with tf.name_scope(self.name) as scope:
            with tf.variable_scope(self.name+"_var", reuse = reuse) as vscope:
                h = tf.layers.conv2d(
                    inputs=h, filters=self.channel, kernel_size=self.filter_size, strides=self.strides,
                    padding=self.padding, use_bias=True,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    data_format='channels_last')
                return tf.nn.relu(h)

# this can use only single gpu
class Conv2D_bn:
    def __init__(self, name, channel,  filter_size=(3, 3), strides=(1, 1), padding='VALID', useBias = False):
#        if re.search("mgpu.py", sys.argv[0]):
#            print("Conv2D_bn cannot use multi gpu enviroment.")
#            exit()
        self.name    = name
        self.channel = channel
        self.filter_size   = filter_size
        self.strides = strides
        self.padding = padding
        self.useBias = useBias

    def batch_norm(self, x, is_training, decay=0.9, eps=1e-5):
        return tf.layers.batch_normalization(
            inputs=x, axis=1 if baseConfig.dataFormat == 'channels_first' else 3,
            momentum=decay, epsilon=eps, center=True,
            scale=True, training=is_training, fused=True)

    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        with tf.name_scope(self.name) as scope:
            with tf.variable_scope(self.name+"_var", reuse = reuse) as vscope:
                h = tf.layers.conv2d(
                    inputs=h, filters=self.channel, kernel_size=self.filter_size, strides=self.strides,
                    padding=self.padding, use_bias=self.useBias,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    data_format=baseConfig.dataFormat)
            with tf.variable_scope(self.name+"_var", reuse = None) as vscope:
                # reuse = Noneでも良いみたい。trainable = Falseだからかもしれない。
                if baseConfig.floatSize == tf.float16:
                    h = tf.cast(h, dtype = tf.float32)
                h = self.batch_norm(h, phaseTrain)
                if baseConfig.floatSize == tf.float16:
                    h = tf.cast(h, dtype = tf.float16)
                return tf.nn.relu(h)

class Flatten:
    def __init__(self, name = "flatten"):
        self.name = name

    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        shape = h.get_shape().as_list()
        channel = shape[1] * shape[2] * shape[3]
        return tf.reshape(h, [-1, channel])

class Dropout:
    def __init__(self, name = "dropout"):
        self.name = name

    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        if baseConfig.floatSize != tf.float32:
            h = tf.cast(h, tf.float32)
        return tf.nn.dropout(h, keepProb)

class FullConnect:
    def __init__(self, name, numClasses, activationProc = tf.nn.softmax):
        self.name = name
        self.numClasses = numClasses
        self.activationProc = activationProc

    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        with tf.name_scope(self.name) as scope:
            with tf.variable_scope(self.name+"_var", reuse = reuse) as vscope:
                h = tf.layers.dense(h, self.numClasses, trainable= not freeze)
                if baseConfig.floatSize == tf.float16:
                    h = tf.cast(h, dtype = tf.float32)
                return self.activationProc(h, name= self.name+"_act")

class Concat:
    def __init__(self, *layers, name = None):
        self.name = name
        self.name = name
        self.layersList = layers
        self.axis = 1 if baseConfig.dataFormat == 'channels_first' else 3

    def apply(self, h, phaseTrain, keepProb, reuse, freeze):
        results = []
        for layers in self.layersList:
            # print("-- concat layer loop ("+self.name+") --")
            h_in = h
            for layer in layers:
                # print(layer.name)
                # print(h_in.shape)
                h_in = layer.apply(h_in, phaseTrain, keepProb, reuse, freeze)
            # print("output")
            # print(h_in.shape)
            results.append(h_in)
        return tf.concat(results, self.axis, name=self.name)

