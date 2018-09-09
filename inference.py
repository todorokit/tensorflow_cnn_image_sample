#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import sys, time
from contextlib import contextmanager

import tensorflow as tf
import cv2
import numpy

import modelcnn
from util.Container import getContainer
from util.utils import *
from util import image as imgModule
from config.classes import classList
from config import baseConfig
import dataset.LargeDataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('config', "config.celeba", 'config module(file) name (no extension).')
flags.DEFINE_string('outdir', 'infout', 'output html name')
flags.DEFINE_integer('batch_size', 2, 'Accuracy batch size. Take care of memory limit.')
flags.DEFINE_float('memory', 0.90, 'Using gpu memory.')
flags.DEFINE_float('other_score', 0.7, '')

Container = getContainer(FLAGS)
config = Container.get("config")

IMAGE_SIZE= config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
NUM_CLASSES = config.NUM_CLASSES

phaseTrain = tf.placeholder(tf.bool, name='phaseTrain')
images_placeholder = tf.placeholder(baseConfig.floatSize, shape=(None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*NUM_RGB_CHANNEL))
labels_placeholder = tf.placeholder(baseConfig.floatSize, shape=(None, NUM_CLASSES))
keepProb = tf.placeholder(baseConfig.floatSize, name="keepProb")

with tf.name_scope("tower_0"):
    logits, _ = modelcnn.inference(images_placeholder, keepProb, config,  False, phaseTrain)
sess = Container.get("sess")
saver = Container.get("saver")

cwd = os.getcwd()

def getImages(path):
    test_image = []
    if config.faceType == "real":
        test_image, test_image_real, faces = imgModule.getFace([path], IMAGE_SIZE)
    elif config.faceType == "anime":
        test_image, test_image_real, faces = imgModule.getAnimeFace([path], IMAGE_SIZE)
    if len(test_image) == 0:
        test_image, test_image_real, faces = imgModule.getImage([path], IMAGE_SIZE)
    return (test_image, test_image_real, faces)

def inferenceAndSave(batch, images):
    arrs = sess.run(logits, feed_dict={images_placeholder: images, phaseTrain: False, keepProb: 1.0})
    i = 0
    for arr in arrs:
        k, real_image, path, dir = batch[i]
        if config.dataType == "multiLabel":
            if isinstance(config.accuracy, tuple):
                method, arg = config.accuracy
                if method == "nth":
                    ans = top1(arr[arg:arg+2]) + arg
                else:
                    raise Exception("config.accuracy is not known")
            else:
                raise Exception("config.accuracy is not known")
        else:
            ans = top1(arr)
        parentDir = os.path.join(cwd, FLAGS.outdir)
        if ans and arr[ans] > FLAGS.other_score:
            destDir = os.path.join(parentDir, classList[ans].replace(" ", "_"))
        else:
            destDir = os.path.join(parentDir, "other")

        file    = os.path.basename(path)
        if dir is None:
            filename, extension = os.path.splitext(file)
        else:
            file_filename, extension = os.path.splitext(file)
            filename = dir + "_" + file_filename
        if k > 0:
            dest = os.path.join(destDir , "%s-%05d%s"% (filename, k, extension))
        else:
            dest = os.path.join(destDir , "%s%s"% (filename,  extension))
        os.makedirs(destDir, exist_ok=True)
        cv2.imwrite(dest, real_image)
        i += 1

def inferenceDir(dir):
    start = time.time()
    batch_len = 0
    batch = []
    batch_img = []
    for file in listDir(dir):
        if os.path.isdir(file):
            for file2 in listDir(file):
                test_image, test_image_real, _ = getImages(file2)
                k = 0
                for img, real in zip(test_image, test_image_real):
                    batch_img.append(img)
                    batch.append((k, real, file2, os.path.basename(file)))
                    k += 1
                    batch_len += 1
                    if batch_len == FLAGS.batch_size:
                        inferenceAndSave(batch, batch_img)
                        batch = []
                        batch_img = []
                        batch_len = 0
        else:
            test_image, test_image_real, _ = getImages(file)
            k = 0
            for img, real in zip(test_image, test_image_real):
                batch_img.append(img)
                batch.append((k, real, file, None))
                k += 1
                batch_len += 1
                if batch_len == FLAGS.batch_size:
                    inferenceAndSave(batch, batch_img)
                    batch = []
                    batch_img = []
                    batch_len = 0

    if batch_len > 0:
        inferenceAndSave(batch, batch_img)
        
def inferenceFile(path):
    test_image, test_image_real, _ = getImages(path)
    test_image = [dataset.LargeDataset.img2vector(path, config, Container)]

    for i in range(len(test_image)):
        image = test_image[i]
#        real_image = cv2.imread(path)
#        cv2.imwrite(os.path.join(cwd, "debug%d.png")% (i), real_image);
        arr = sess.run(logits, feed_dict={images_placeholder: [image], phaseTrain:False, keepProb: 1.0})[0]
        if config.dataType == "multi-label":
            indices = sess.run(tf.nn.top_k(arr, 10).indices)
        else:
            indices = top5(arr)
        print ("----- %02d ----" % (i))
        for j in indices:
            print("%s %g" %(classList[j] , arr[j]))
        
def main(args):
    if len(args) == 1:
        print(args[0]+" filename_or_dirname")
        exit()
    if os.path.isdir(args[1]):
        inferenceDir(args[1])
    else:
        inferenceFile(args[1])

tf.app.run()
