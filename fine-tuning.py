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

import deeptool, modelcnn

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('config', "config", 'config module(file) name (no extension).')
config = __import__(FLAGS.config)

WSCALE = config.WSCALE

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
conv2dList=config.conv2dList
modelFile = config.modelFile

cwd = os.getcwd()
modelpath = os.path.join(cwd, modelFile)

if __name__ == '__main__':
    ## --------------------------------
    # backup
    now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
    deeptool.backup(FLAGS.config+".py", modelFile, "backup", now)

    keep_prob = tf.placeholder("float")

    ## --------------------------------
    ## restore
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    phaseTrain = tf.placeholder(tf.bool, name='phase_train')

    # tuneArrayにNUM_CLASSESに関わる変数は入らない。
    logits, tuneArray = modelcnn.inference(images_placeholder, keep_prob, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, WSCALE, False, phaseTrain)
    loss_value = modelcnn.loss(logits, labels_placeholder)
    train_op = modelcnn.training(loss_value, 1e-4)

    sess = tf.Session()
    saver = tf.train.Saver(tuneArray)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, modelpath)

    ## --------------------------------
    ## SAVE
    train_image, train_label, _ = deeptool.loadImages(FLAGS.train, IMAGE_SIZE, NUM_CLASSES)
    
    saver = tf.train.Saver()
    sess.run(train_op, feed_dict={
        images_placeholder: train_image[0:1],
        labels_placeholder: train_label[0:1],
        keep_prob: 0.5,
        phaseTrain: True
    })
    save_path = saver.save(sess, modelpath)
