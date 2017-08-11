#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os, sys

import tensorflow as tf
import cv2

import modelcnn
from util.Container import Container
from util.utils import *
from util import image as imgModule
from config.classes import classList

def main(args):
    config = Container.get("config")
    IMAGE_SIZE= config.IMAGE_SIZE
    NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
    NUM_CLASSES = config.NUM_CLASSES

    test_image, test_image_real, _ = imgModule.getAnimeFace(args[1:], IMAGE_SIZE)
    if (len(test_image) == 0 ):
        test_image, test_image_real, _ = imgModule.getFace(args[1:], IMAGE_SIZE)
    if (len(test_image) == 0 ):
        test_image, test_image_real, _ = imgModule.getImage(args[1:], IMAGE_SIZE)

    keep_prob = tf.constant(1.0)
    phaseTrain = tf.placeholder(tf.bool, name='phase_train')

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    
    logits, _ = modelcnn.inference(images_placeholder, keep_prob, config,  False, phaseTrain)
    
    sess = Container.get("sess")
    saver = Container.get("saver")
    cwd = os.getcwd()
    
    for i in range(len(test_image)):
        image = test_image[i]
        real_image = test_image_real[i]
    #        cv2.imwrite(os.path.join(cwd, "debug%d.png")% (i), real_image);
        arr = sess.run(logits, feed_dict={images_placeholder: [image], phaseTrain:False})[0]
        if config.dataType == "multiLabel":
            indices = sess.run(tf.nn.top_k(arr, len(config.NUM_CLASSES_LIST)).indices)
        else:
            indices = top5(arr)
        print ("----- %02d ----" % (i))
        for j in indices:
            print("%s %g" %(classList[j] , arr[j]))

tf.app.run()
