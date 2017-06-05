#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, time, math
from collections import OrderedDict

import tensorflow as tf
import tensorflow.python.platform

import config, modelcnn, deeptool

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
WSCALE = config.WSCALE

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', 'c:\\tmp\\image_cnn', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 3, 'Number of steps to run trainer.')
flags.DEFINE_string('batch_size', 20, 'Batch size.Must divide evenly into the dataset sizes and config.num_gpu.')
flags.DEFINE_integer('acc_batch_size', 5000, 'Accuracy batch size. This must divide evenly into the test data set sizes.')
flags.DEFINE_float('learning_rate_base', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('learning_rate_odds', 0.9, 'Initial learning rate * this rate.')
flags.DEFINE_float('num_loop', 5, 'num loop .')

conv2dList = config.conv2dList

# 変数名が被らないようにすれば、各GPUで全く別のlearningができそう。でも断念。
if __name__ == '__main__':
    train_image, train_label, _ = deeptool.loadImages(FLAGS.train, IMAGE_SIZE, NUM_CLASSES)
    result = OrderedDict()
    filterSizes = [15,18,21]
    channels = [28,32,36]

    for params in [(FLAGS.learning_rate_base * (FLAGS.learning_rate_odds** loop),
                    filterSize,
                    channel,
                    )
                   for loop in range(FLAGS.num_loop)
                   for filterSize in filterSizes
                   for channel in channels
                  ]:
        learningRate, filterSize, channel = params
        
        conv2dList[0] = ("conv1", filterSize, channel)
        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            phs = modelcnn.Placeholders(IMAGE_SIZE, IMAGE_SIZE, NUM_RGB_CHANNEL, NUM_CLASSES)
            dataset = modelcnn.InMemoryDataset(train_image, train_label, [], [], FLAGS.batch_size, FLAGS.acc_batch_size)
            train_op, acc_op, _, debug = modelcnn.multiGpuLearning(learningRate, phs, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, NUM_CLASSES, WSCALE)

            sess.run(tf.global_variables_initializer())
            
            # training
            n = int(len(train_image)/FLAGS.batch_size)
            for step in range(FLAGS.max_steps):
                startTime = time.time()
                for i in dataset.getTrainLoop():
                    sess.run(train_op, feed_dict=phs.getDict(
                        dataset.getTrainImage(i),
                        dataset.getTrainLabel(i),
                        0.5
                    ))
            accuracy = modelcnn.calcAccuracy(sess, acc_op, phs, dataset)

            fp = open("search-result.txt", "a")
            fom = "test params %s, accuracy %g\n"
            fp.write(fom%(params, accuracy))
            fp.close()
