import tensorflow as tf
import tensorflow.python.platform
import re

import config

def inference(images_placeholder, imageSize, numInitChannel, conv2dList, fc1Channel, numClasses,wscale, keep_prob):
    def weight_variable(shape, wscale= 0.1):
        
#      print (shape)
      initial = tf.truncated_normal(shape, stddev=wscale)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x, size = 2, slide = 2):
      return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                            strides=[1, slide, slide, 1], padding='SAME')
    
    def max_pool(name, x, size = 2):
        with tf.name_scope(name) as scope:
            return max_pool_2x2(x, size)


    def conv2dWeightBias(name, x, inChannel, outChannel, filterSize, slide = 1, wscale=1):
        with tf.name_scope(name) as scope:
            W = weight_variable([filterSize, filterSize, inChannel, outChannel], wscale)
            b = bias_variable([outChannel])
            return tf.nn.conv2d(x, W, strides=[1, slide, slide, 1], padding='SAME')+ b

    def linear(name, x, inChannel, outChannel):
        with tf.name_scope(name) as scope:
            W = weight_variable([inChannel, outChannel])
            b = bias_variable([outChannel])
            return tf.matmul(h, W) + b
        
    prevChannel = numInitChannel
    h = tf.reshape(images_placeholder, [-1, imageSize, imageSize, numInitChannel])
    for name, filterSize, channel in conv2dList:
        if (re.search("^pool", name)):
            h = max_pool(name, h, filterSize)
            imageSize = imageSize // 2
        else:
            h = tf.nn.relu(conv2dWeightBias(name,h , prevChannel, channel, filterSize, 1, wscale))
            h = tf.nn.relu(conv2dWeightBias(name + "-a", h, channel, channel, 1, 1, wscale))
            h = tf.nn.relu(conv2dWeightBias(name +"-b", h, channel, channel, 1, 1, wscale))
            prevChannel = channel
    
    prevChannel *= imageSize * imageSize
    h = tf.reshape(h, [-1,  prevChannel])
    h = tf.nn.relu(linear(name, h, prevChannel, fc1Channel))
    h = tf.nn.dropout(h, keep_prob)
    y = tf.nn.softmax(linear("fc2", h, fc1Channel, numClasses))

    return y

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy
