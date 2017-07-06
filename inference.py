#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os, sys
from contextlib import contextmanager

import tensorflow as tf
import cv2

import config2
import deeptool, modelcnn

flags = tf.app.flags
FLAGS = flags.FLAGS

# 速いけど、gpuの邪魔したくない
flags.DEFINE_integer('use_cpu', 0, 'using cpu')
flags.DEFINE_string('config', "config", 'config module(file) name (no extension).')
config = __import__(FLAGS.config)

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
conv2dList=config.conv2dList
wscale = config.WSCALE
phaseTrain = tf.placeholder(tf.bool, name='phase_train')

def top5(arr):
    return arr.argsort()[-5:][::-1]

@contextmanager
def WithNone():
    yield

# multi gpu 化する意味は全くない。
def main(args):
    test_image, test_image_real, _ = deeptool.getAnimeFace(args[1:], IMAGE_SIZE)
    if (len(test_image) == 0 ):
        test_image, test_image_real, _ = deeptool.getFace(args[1:], IMAGE_SIZE)
    if (len(test_image) == 0 ):
        test_image, test_image_real, _ = deeptool.getImage(args[1:], IMAGE_SIZE)

    keep_prob = tf.constant(1.0)
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    
    with WithNone() if FLAGS.use_cpu == 0 else tf.device("/cpu:0"):
        logits, _ = modelcnn.inference(images_placeholder, keep_prob, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, wscale, False, phaseTrain)
    sess = tf.InteractiveSession()
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    cwd = os.getcwd()
    saver.restore(sess, os.path.join(cwd, config.modelFile))
    
    for i in range(len(test_image)):
        image = test_image[i]
        real_image = test_image_real[i]
    #        cv2.imwrite(os.path.join(cwd, "debug%d.png")% (i), real_image);
        arr = logits.eval(feed_dict={images_placeholder: [image], phaseTrain:False})[0]
        indices = top5(arr)
        print ("----- %02d ----" % (i))
        for j in indices:
            print("%s %g" %(config2.classList[j] , arr[j]))

tf.app.run()
                
