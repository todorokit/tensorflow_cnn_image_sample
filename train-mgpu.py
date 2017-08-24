#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time

import tensorflow as tf
import tensorflow.python.platform

from util.Container import Container
from util.utils import *
from util.MyTimer import MyTimer

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epoch', 1000, 'Number of epoch to run trainer.')
flags.DEFINE_integer('batch_size', 80, 'Training batch size. This must divide evenly into the train dataset sizes and config.num_gpu.')
flags.DEFINE_integer('acc_batch_size', 80, 'Accuracy batch size. This must divide evenly into the test data set sizes.')

printCUDA_env()

def main(_):
    with tf.Graph().as_default():
        config = Container.get("config")
        phs = Container.get("placeholders")
        train_op, acc_op = Container.get("ops_mgpu")
        sess = Container.get("sess")
        saver = Container.get("saver")

        trainDataset = Container.get("traindataset")
        testDataset =  Container.get("testdataset")
        validDataset =  Container.get("validdataset")

        for epoch in range(FLAGS.epoch):
            timer = MyTimer()
            trainDataset.train(sess, train_op, phs)
            # train_accuracy = trainDataset.calcAccuracy(sess, acc_op, phs)
            test_accuracy = testDataset.calcAccuracy(sess, acc_op, phs)
            valid_accuracy = 0
            if validDataset is not None:
                valid_accuracy = validDataset.calcAccuracy(sess, acc_op, phs)
            saveBest(config, FLAGS, sess, saver, test_accuracy)
            # train_accuracy
            print("%s: epoch %d, (%g, %g), %g data/sec"%(timer.getNow("%H:%M:%S"), epoch, test_accuracy, valid_accuracy, trainDataset.getLen()/timer.getTime()))
            sys.stdout.flush()
            saver.save()

tf.app.run()
