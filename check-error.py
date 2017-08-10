#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import tensorflow.python.platform

import modelcnn
from util.Container import Container
from util.utils import *
from util import image as imgModule
from config.classes import classList

config = Container.get("config")
NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL

flags = tf.app.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
    test_image, test_label, paths =  imgModule.loadImages(config.testFile, IMAGE_SIZE, NUM_CLASSES)
    
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")
    phaseTrain = tf.placeholder(tf.bool, name='phase_train')

    logits, _ = modelcnn.inference(images_placeholder, keep_prob, config, False, phaseTrain)

    sess = Container.get("sess")
    saver = Container.get("saver")
    cwd = os.getcwd()

    oks = []
    lowscores = []
    ngs = []
    stat = {}
    for image, label, path in zip(test_image, test_label, paths):
        arr = sess.run(logits, feed_dict={images_placeholder: [image],keep_prob: 1.0, phaseTrain: False})[0]
        labelVal = top1(label)
        topVal = top1(arr)
        score = arr[topVal]
        if (topVal == labelVal):
            if ( score < 0.5) :
                lowscores.append((path ,classList[topVal], score))
            else:
                oks.append((path ,classList[topVal], score))
        else:
            ngs.append((path, classList[labelVal], classList[topVal], score))
            try:
                stat[labelVal] = stat[labelVal] + 1
            except:
                stat[labelVal] = 1

    i = 0
    tds = []
    trs = []
    def img (src):
        return "<img width='25%%' src='file:///%s'/>" % (os.path.join(cwd, path).replace("\\", "/"))
    
    for ng in ngs :
        path , labelName, className, score = ng
        i+=1
        tds.append("<td>%s<br/>%s:%s<br/>%g</td>\n" % (img(path), labelName, className, score))
        if (i >= 4):
            trs.append("<tr>"+"".join(tds)+"</tr>")
            tds = []
            i = 0
    ngstr = "".join(trs)
            
    i = 0
    tds = []
    trs = []
    for low in lowscores :
        path , labelName, score = low
        i+=1
        tds.append("<td>%s<br/>%s<br/>%g</td>\n" % (img(path), labelName, score))
        if (i >= 4):
            trs.append("<tr>"+"".join(tds)+"</tr>")
            tds = []
            i = 0
    lowstr = "".join(trs)

    trs = []
    for label in stat:
        trs.append("<tr><td>%s</td><td>%d</td></tr>" % (classList[label], stat[label]))
    statstr = "".join(trs)
    print("""
    <html><body>
    STAT<br>
    <table border='1'>%s</table>
    MISTAKEN<br>
    <table border='1'>%s</table>
    LOW SCORES<br>
    <table border='1'>%s</table>
    </body></html>""" % (statstr,ngstr, lowstr))
