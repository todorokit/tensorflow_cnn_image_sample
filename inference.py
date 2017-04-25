#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

import config
import config2
import deeptool
import modelcnn

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL

def top5(arr):
    return arr.argsort()[-5:][::-1]

if __name__ == '__main__':
    # アニメ画像の顔を取得 "lbpcascade_animeface.xml"が必要
    test_image = deeptool.getAnimeFace(sys.argv[1:], IMAGE_SIZE)

    keep_prob = tf.constant(1.0)
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))

    logits = modelcnn.inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    cwd = os.getcwd()
    saver.restore(sess, cwd+"\\model.ckpt")

    for image in test_image:
        arr = logits.eval(feed_dict={images_placeholder: [image]})[0]
        indices = top5(arr)
        for j in indices:
            print("%s %f" %(config2.classList[j] , arr[j]))
