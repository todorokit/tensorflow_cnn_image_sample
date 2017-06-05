#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os, sys

import tensorflow as tf
import cv2

import config, config2
import deeptool, modelcnn

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
conv2dList=config.conv2dList
wscale = config.WSCALE

def top5(arr):
    return arr.argsort()[-5:][::-1]

# multi gpu 化する意味は全くない。
if __name__ == '__main__':
    test_image, test_image_real, _ = deeptool.getAnimeFace(sys.argv[1:], IMAGE_SIZE)

    keep_prob = tf.constant(1.0)
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))

    logits, _ = modelcnn.inference(images_placeholder, keep_prob, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, NUM_CLASSES, wscale, False)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    cwd = os.getcwd()
    saver.restore(sess, os.path.join(cwd, config.modelFile))

    for i in range(len(test_image)):
        image = test_image[i]
        real_image = test_image_real[i]
#        cv2.imwrite(os.path.join(cwd, "debug%d.png")% (i), real_image);
        arr = logits.eval(feed_dict={images_placeholder: [image]})[0]
        indices = top5(arr)
        print ("----- %02d ----" % (i))
        for j in indices:
            print("%s %g" %(config2.classList[j] , arr[j]))
