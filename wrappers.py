from pprint import pprint

import tensorflow as tf
import tensorflow.python.platform

class Layer():
    # 1 gpu の場合、cpuに保存しない方が速い。multi gpu の場合, cpu側のメモリに保存しなくてはならない
    def makeVar(self, name, shape, initializer, trainable=True):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
            return var

    def weight_variable(self, tuneArray, shape, wscale= 0.1):
        #      print (shape, wscale)
        v = self.makeVar("W", shape, initializer=tf.truncated_normal_initializer(stddev=wscale))
        if (tuneArray is not None):
            tuneArray.append(v)
        return v

    def bias_variable(self, tuneArray, shape):
        v = self.makeVar("b", shape, initializer=tf.constant_initializer(0.1))
        if (tuneArray is not None):
            tuneArray.append(v)
        return v

class MaxPooling2D(Layer):
    def __init__(self, name, pool_size=(3, 3), strides=(2, 2), padding='VALID'):
        self.name    = name
        self.ksize   = [1, pool_size[0], pool_size[1], 1]
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        with tf.name_scope(self.name) as scope:
            return tf.nn.max_pool(h, ksize=self.ksize, strides=self.strides, padding=self.padding)

class Conv2D_bn(Layer):
    def __init__(self, name, channel,  filter_size=(3, 3), strides=(1, 1), padding='VALID'):
        self.name    = name
        self.channel = channel
        self.filter_size   = filter_size
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding

    def batch_norm_wrapper(self, tuneArray, inputs, phase_train=None, decay=0.99):
        epsilon = 1e-5
        out_dim = inputs.get_shape()[-1]
        beta = self.makeVar("beta", [out_dim], initializer=tf.zeros_initializer())
        gamma = self.makeVar("gamma", [out_dim], initializer=tf.ones_initializer())
# 学習と推論で完全に別のロジックが動くと問題あり。
#        if phase_train is None:
#            return tf.nn.batch_normalization(inputs, tf.zeros([out_dim]), tf.ones([out_dim]), beta, gamma, epsilon)
        batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2])
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        with tf.name_scope(self.name) as scope:
            with tf.variable_scope(self.name+"_var", reuse = reuse) as vscope:
                prevChannel = h.shape[3]
                W = self.weight_variable(tuneArray, [self.filter_size[0], self.filter_size[1], prevChannel, self.channel], wscale)
#                b = self.bias_variable(tuneArray, [self.channel])
                h = tf.nn.conv2d(h, W, strides=self.strides, padding=self.padding)
                h = self.batch_norm_wrapper(tuneArray, h, phaseTrain)
                return tf.nn.relu(h)

class Flatten(Layer):
    def __init__(self, name = "flatten"):
        self.name = name

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        shape = h.get_shape().as_list()
        channel = shape[1] * shape[2] * shape[3]
        return tf.reshape(h, [-1, channel])

class Dropout(Layer):
    def __init__(self, name = "dropout"):
        self.name = name

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        return tf.nn.dropout(h, keepProb)

class Lrn(Layer):
    def __init__(self, name = "lrn"):
        self.name = name

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        return tf.nn.lrn(h)

class FullConnect(Layer):
    def __init__(self, name, numClasses, activationProc = tf.nn.softmax):
        self.name = name
        self.numClasses = numClasses
        self.activationProc = activationProc

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        with tf.name_scope(self.name) as scope:
            with tf.variable_scope(self.name+"_var", reuse = reuse) as vscope:
                prevChannel = h.shape[1]
                W = self.weight_variable(None, [prevChannel, self.numClasses], wscale)
                b = self.bias_variable(None, [self.numClasses])
                return self.activationProc(tf.matmul(h, W) + b)
