#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import math
from collections import OrderedDict

import tensorflow as tf
import tensorflow.python.platform

import config
import modelcnn
import deeptool

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
wscale = config.WSCALE

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', 'c:\\tmp\\image_cnn', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 3, 'Number of steps to run trainer.')
flags.DEFINE_string('batch_size', 10, 'Batch size'
                    'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate_base', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('learning_rate_odds', 0.9, 'Initial learning rate.')
flags.DEFINE_float('num_loop', 5, 'Initial learning rate.')

conv2dList = config.conv2dList

if __name__ == '__main__':
    train_image, train_label, _ = deeptool.loadImages(FLAGS.train, IMAGE_SIZE, NUM_CLASSES)
#    test_image, test_label, _ =  deeptool.loadImages(FLAGS.test, IMAGE_SIZE, NUM_CLASSES)
#    batch_sizes = [ int(sizestr) for sizestr in FLAGS.batch_sizes.split(",")]
    batch_size = FLAGS.batch_size
    result = OrderedDict()
    filterSizes = [18]
    channels = [32]
    fcChannels = [1200, 1400, 1500, 1600, 1800]
#    wscales = [math.sqrt(2.0/NUM_CLASSES)]

    for params in [(FLAGS.learning_rate_base * (FLAGS.learning_rate_odds** loop),
#                    batch_size,
                    filterSize,
                    channel,
                    fcChannel
#                    wscale
                    )
                   for loop in range(FLAGS.num_loop)
#                   for batch_size in batch_sizes
                   for filterSize in filterSizes
                   for channel in channels
                   for fcChannel in fcChannels
#                   for wscale in wscales
                  ]:
#        learningRate, batch_size, filterSize, channel, fcChannel, wscale = params
        learningRate, filterSize, channel, fcChannel = params
        conv2dList[0] = ("conv1", filterSize, channel)
        with tf.Graph().as_default(), tf.Session() as sess:
            timeStart = time.time()
            images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL))
            labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
            keep_prob = tf.placeholder("float")
            
            logits = modelcnn.inference(images_placeholder, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, fcChannel, NUM_CLASSES, wscale, keep_prob)
            loss_value = modelcnn.loss(logits, labels_placeholder)
            acc = modelcnn.accuracy(logits, labels_placeholder)

            train_op = modelcnn.training(loss_value, learningRate)

            sess.run(tf.global_variables_initializer())
            
            feedDictNoProb = {
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0}
            
            # 訓練の実行
            n = int(len(train_image)/batch_size)
            for step in range(FLAGS.max_steps):
                for i in range(n):
                    batch = batch_size*i
                    sess.run(train_op, feed_dict={
                        images_placeholder: train_image[batch:batch+batch_size],
                        labels_placeholder: train_label[batch:batch+batch_size],
                        keep_prob: 0.50})
                
            timeTrain = time.time()
            accVal = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})

            fom = "test params %s, accuracy %g"
            print(fom%(params, accVal))
            sys.stdout.flush()
