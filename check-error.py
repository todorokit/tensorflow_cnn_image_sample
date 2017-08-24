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
from config import baseConfig

config = Container.get("config")
NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('outfile', 'miss.html', 'output html name')
flags.DEFINE_integer('acc_batch_size', 80, 'Accuracy batch size. Take care of memory limit.')

def main(_):
    testDataset =  Container.get("testdataset")
    
    images_placeholder = tf.placeholder(baseConfig.floatSize, shape=(None, IMAGE_SIZE[0]*IMAGE_SIZE[1]*NUM_RGB_CHANNEL))
    labels_placeholder = tf.placeholder(baseConfig.floatSize, shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder(baseConfig.floatSize)
    phaseTrain = tf.placeholder(tf.bool, name='phase_train')

    with tf.name_scope("tower_0"):
        logits, _ = modelcnn.inference(images_placeholder, keep_prob, config, False, phaseTrain)

    sess = Container.get("sess")
    saver = Container.get("saver")
    cwd = os.getcwd()

    oks = []
    lowscores = []
    ngs = []
    stat = {}
    arg = 0
    for images, labels, paths in testDataset.flow():
        ix = 0
        arrs = sess.run(logits, feed_dict={images_placeholder: images,keep_prob: 1.0, phaseTrain: False})
        for arr in arrs:
            if config.dataType == "multiLabel":
                if isinstance(config.accuracy, tuple):
                    method, arg = config.accuracy
                    if method == "nth":
                        labelVal = top1(labels[ix][arg:arg+2]) + arg
                        topVal = top1(arr[arg:arg+2]) + arg
                    else:
                        raise Exception("config.accuracy is not known")
                else:
                    raise Exception("config.accuracy is not known")
            else:
                labelVal = top1(labels[ix])
                topVal = top1(arr)
            score = arr[topVal]
            if (topVal == labelVal):
                if ( score < 0.5) :
                    lowscores.append((paths[ix], classList[topVal], score))
                else:
                    oks.append((paths[ix], classList[topVal], score))
            else:
                ngs.append((paths[ix], classList[labelVal], classList[topVal], score))
                try:
                    stat[labelVal] = stat[labelVal] + 1
                except:
                    stat[labelVal] = 1
            ix += 1

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

    fp = open(FLAGS.outfile, "w")
    fp.write("""
        <html><body>
        STAT<br>
        <table border='1'>%s</table>
        MISTAKEN<br>
        <table border='1'>%s</table>
        LOW SCORES<br>
        <table border='1'>%s</table>
        </body></html>""" % (statstr,ngstr, lowstr))

tf.app.run()
