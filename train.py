#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

import tensorflow as tf

from util.Container import getContainer
from util.utils import *
from util.MyTimer import MyTimer

flags = tf.app.flags
flags.DEFINE_integer('epoch', 1000, 'Number of epoch to run trainer.')
flags.DEFINE_integer('batch_size', 80, 'Training batch size. This must divide evenly into the train dataset sizes.')
flags.DEFINE_integer('acc_batch_size', 500, 'Accuracy batch size. Take care of memory limit.')
flags.DEFINE_string('config', "config.celeba", 'config module(file) name (no extension).')
flags.DEFINE_float('memory', 0.90, 'Using gpu memory.')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
FLAGS = flags.FLAGS

def main(argv):
    with tf.Graph().as_default():

        Container = getContainer(FLAGS)
        config = Container.get("config")
        phs = Container.get("placeholders")
        if config.num_gpu > 1 :
            gpumode = "MULTI GPU MODE"
            train_op, acc_op = Container.get("ops_mgpu")
        else:
            gpumode = "SINGLE GPU MODE"
            train_op, acc_op = Container.get("ops")
        print ({"gpumode": gpumode, "learning_rate": FLAGS.learning_rate})
        sess = Container.get("sess")
        saver = Container.get("saver")

        if config.dataType == "multi-label":
            trainDataset = Container.get("multilargetraindataset")
            testDataset =  Container.get("multilargetestdataset")
            validDataset = None
        else:
            trainDataset = Container.get("largetraindataset")
            testDataset = Container.get("largetestdataset")
            validDataset = None

        for epoch in range(FLAGS.epoch):
            timer = MyTimer()
            test_accuracy = -1.0
            valid_accuracy = -1.0
            trainDataset.train(sess, train_op, phs)
            test_accuracy = testDataset.calcAccuracy(sess, acc_op, phs)
            valid_accuracy = -1.0
            if validDataset is not None:
                valid_accuracy = validDataset.calcAccuracy(sess, acc_op, phs)
            
            saveBest(config, FLAGS, sess, saver, test_accuracy)
            saver.save()

            # train_accuracy, 
            print("%s: epoch %d, (%g, %g) %g data/sec"%(timer.getNow("%H:%M:%S"), epoch, test_accuracy, valid_accuracy , trainDataset.getLen() / timer.getTime()))
            sys.stdout.flush()

if __name__ == '__main__':
    tf.app.run()
