import sys, re
from pprint import pprint

import tensorflow as tf
import tensorflow.python.platform

# name を このclassに持ってくる。
class Layer(object):
    
    # 1 gpu の場合、cpuに保存しない方が速い。multi gpu の場合, cpu側のメモリに保存しなくてはならない
    def makeVar(self, name, shape, initializer, trainable=True):
        # FIXME: darty hack . Use DIContainer.
        if re.search("mgpu.py", sys.argv[0]):
            with tf.device('/cpu:0'):
                var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
        else:
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

class Pooling2D(Layer):
    def __init__(self, name, pool_size=(3, 3), strides=(2, 2), padding='VALID'):
        self.name    = name
        self.ksize   = [1, pool_size[0], pool_size[1], 1]
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding
    
class MaxPooling2D(Pooling2D):
    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        with tf.name_scope(self.name) as scope:
            return tf.nn.max_pool(h, ksize=self.ksize, strides=self.strides, padding=self.padding)

class AveragePooling2D(Pooling2D):
    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        with tf.name_scope(self.name) as scope:
            return tf.nn.avg_pool(h, ksize=self.ksize, strides=self.strides, padding=self.padding)

class GlobalAveragePooling2D(Layer):
    def __init__(self, name, data_format="channels_last"):
        self.name = name
        self.data_format = data_format

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        if self.data_format == 'channels_last':
            return tf.reduce_mean(h, reduction_indices=[1, 2])
        else:
            return tf.reduce_mean(h, reduction_indices=[2, 3])

class Conv2D(Layer):
    def __init__(self, name, channel,  filter_size=(3, 3), strides=(1, 1), padding='VALID'):
        self.name    = name
        self.channel = channel
        self.filter_size   = filter_size
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        with tf.name_scope(self.name) as scope:
            with tf.variable_scope(self.name+"_var", reuse = reuse) as vscope:
                prevChannel = h.shape[3]
                W = self.weight_variable(tuneArray, [self.filter_size[0], self.filter_size[1], prevChannel, self.channel], wscale)
                b = self.bias_variable(tuneArray, [self.channel])
                h = tf.nn.conv2d(h, W, strides=self.strides, padding=self.padding) + b
                return tf.nn.relu(h)

# this can use only single gpu
class Conv2D_bn(Layer):
    def __init__(self, name, channel,  filter_size=(3, 3), strides=(1, 1), padding='VALID', useBias = False):
        if re.search("mgpu.py", sys.argv[0]):
            print("Conv2D_bn cannot use multi gpu enviroment.")
            exit()
        self.name    = name
        self.channel = channel
        self.filter_size   = filter_size
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding
        self.useBias = useBias

    def batch_norm(self, tuneArray, x, is_training, decay=0.9, eps=1e-5):
      shape = x.get_shape().as_list()
      assert len(shape) in [2, 4]
    
      n_out = shape[-1]
      beta = tf.Variable(tf.zeros([n_out]))
      gamma = tf.Variable(tf.ones([n_out]))
    
      if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
      else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    
      ema = tf.train.ExponentialMovingAverage(decay=decay)
    
      def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
          return tf.identity(batch_mean), tf.identity(batch_var)
      mean, var = tf.cond(is_training, mean_var_with_update,
                          lambda : (ema.average(batch_mean), ema.average(batch_var)))
      return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        with tf.name_scope(self.name) as scope:
            with tf.variable_scope(self.name+"_var", reuse = reuse) as vscope:
                prevChannel = h.shape[3]
                W = self.weight_variable(tuneArray, [self.filter_size[0], self.filter_size[1], prevChannel, self.channel], wscale)
                if self.useBias:
                    b = self.bias_variable(tuneArray, [self.channel])
                    h = tf.nn.conv2d(h, W, strides=self.strides, padding=self.padding) + b
                else:
                    h = tf.nn.conv2d(h, W, strides=self.strides, padding=self.padding)
                h = self.batch_norm(tuneArray, h, phaseTrain)
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

class Concat(Layer):
    def __init__(self, *layers, name = None, axis = 3):
        self.name = name
        self.layersList = layers
        self.axis = axis

    def apply(self, tuneArray, h, wscale, phaseTrain, keepProb, reuse):
        results = []
        for layers in self.layersList:
            # print("-- concat layer loop ("+self.name+") --")
            h_in = h
            for layer in layers:
                # print(layer.name)
                # print(h_in.shape)
                h_in = layer.apply(tuneArray, h_in, wscale, phaseTrain, keepProb, reuse)
            # print("output")
            # print(h_in.shape)
            results.append(h_in)
        return tf.concat(results, self.axis, name=self.name)
