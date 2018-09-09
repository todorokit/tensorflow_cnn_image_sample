#!/usr/bin/env python
#! -*- coding: utf-8 -*-

## NUM_CLASSES のみが変わる前提で組まれています。
## config.pyを変えた後に使用して下さい

## MEMO
## モデル(logits)を作り直してはいけない。=モデルと同じ変数でなければいけない。
## runしないと本trainで取り込めない。

## MEMO
## multi GPU でもモデルは変わらない

import os, sys, os.path, datetime

import tensorflow as tf

import modelcnn
from util.Container import getContainer
from util.utils import *
from util.MyTimer import MyTimer
from util import image as imgModule
from config import baseConfig

config = Container.get("config")
FLAGS = Container.get("flags")
NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
modelFile = config.modelFile

cwd = os.getcwd()
modelpath = os.path.join(cwd, modelFile)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('config', "config.celeba", 'config module(file) name (no extension).')
flags.DEFINE_integer('epoch', 1000, 'Number of epoch to run trainer.')
flags.DEFINE_integer('batch_size', 80, 'Training batch size. This must divide evenly into the train dataset sizes.')
flags.DEFINE_integer('acc_batch_size', 500, 'Accuracy batch size. Take care of memory limit.')
flags.DEFINE_float('memory', 0.90, 'Using gpu memory.')
flags.DEFINE_boolean('freeze', True, 'Number of epoch to run trainer.')

def main(_):
    with tf.Graph().as_default():
        ## --------------------------------
        # backup
        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        backup(deepImportToPath(FLAGS.config), modelFile, "backup", now)

        ## --------------------------------
        ## restore
        Container = getContainer(FLAGS)
        phs = Container.get("placeholders")
        keep_prob = phs.getKeepProb()
        images_placeholder = phs.getImages()
        labels_placeholder = phs.getLabels()
        phaseTrain = phs.getPhaseTrain()

        # tuneArrayにFC変数は入らない。その他 EMAの変数等も入らない
        with tf.name_scope("tower_0"):
            logits, tuneArray = modelcnn.inference(images_placeholder, keep_prob, config, False, phaseTrain, FLAGS.freeze)
        loss_value = modelcnn.loss(logits, labels_placeholder)
        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss_value)
        acc_op = modelcnn.getAccuracyOp(logits, labels_placeholder, config)

        sess = tf.Session()
        saver = tf.train.Saver([v for v in tf.global_variables() if not v.name.startswith("fc")])
        sess.run(tf.variables_initializer([v for v in tf.global_variables() if v.name.startswith("fc")]))
        saver.restore(sess, modelpath)

        trainDataset = Container.get("traindataset")
        testDataset =  Container.get("testdataset")
        validDataset =  Container.get("validdataset")
        # 全ての変数を保存するsaverを用意する
        mysaver =  Container.get("saver_no_restore")

        for epoch in range(FLAGS.epoch):
            timer = MyTimer()
            trainDataset.train(sess, train_op, phs)
            test_accuracy = testDataset.calcAccuracy(sess, acc_op, phs)
            valid_accuracy = 0
            if validDataset is not None:
                valid_accuracy = validDataset.calcAccuracy(sess, acc_op, phs)
            
            saveBest(config, FLAGS, sess, mysaver, test_accuracy)

            print("%s: epoch %d, (%g, %g), %g data/sec"%(timer.getNow("%H:%M:%S"), epoch, test_accuracy, valid_accuracy, trainDataset.getLen()/timer.getTime()))
            sys.stdout.flush()
            mysaver.save()

tf.app.run()
