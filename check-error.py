#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import tensorflow.python.platform

import config
import config2
import deeptool
import modelcnn

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', 'c:\\tmp\\image_cnn', 'Directory to put the training data.')

def top1(arr):
    return arr.argsort()[-1:][::-1]

if __name__ == '__main__':
    test_image, test_label, paths =  deeptool.loadImages(FLAGS.test, IMAGE_SIZE, NUM_CLASSES)
    
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = modelcnn.inference(images_placeholder, keep_prob)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    cwd = os.getcwd()
    saver.restore(sess, cwd+"\\model.ckpt")

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    for image, label, path in zip(test_image, test_label, paths):
        arr = sess.run(logits, feed_dict={images_placeholder: [image],keep_prob: 1.0})[0]
        indices = top1(arr)
        labelVal = top1(label)[0]
        topVal = indices[0]
        if (topVal == labelVal):
            print("ok %s %g %s" %(config2.classList[topVal] , arr[topVal], path))
        else:
            print("ng label:%s result:%s %g %s" %(config2.classList[labelVal], config2.classList[topVal] , arr[topVal], path))
