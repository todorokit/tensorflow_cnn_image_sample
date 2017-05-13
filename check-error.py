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
conv2dList=config.conv2dList
FC_CHANNEL = config.FC_CHANNEL
wscale = config.WSCALE

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

    logits = modelcnn.inference(images_placeholder, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, FC_CHANNEL, NUM_CLASSES, wscale, keep_prob)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    cwd = os.getcwd()
    saver.restore(sess, cwd+"\\model.ckpt")

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    oks = []
    lowscores = []
    ngs = []
    for image, label, path in zip(test_image, test_label, paths):
        arr = sess.run(logits, feed_dict={images_placeholder: [image],keep_prob: 1.0})[0]
        indices = top1(arr)
        labelVal = top1(label)[0]
        topVal = indices[0]
        score = arr[topVal]
        if (topVal == labelVal):
            if ( score < 0.5) :
                lowscores.append((path ,config2.classList[topVal], score))
            else:
                oks.append((path ,config2.classList[topVal], score))
        else:
            ngs.append((path, config2.classList[labelVal], config2.classList[topVal], score))

    i = 0
    tds = []
    trs = []
    def img (src):
        return "<img width='25%%' src='file:///%s'/>" % (path.replace("\\", "/"))
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
    print("""
    <html><body>
    間違ったもの<br>
    <table border='1'>%s</table>
    スコアが低いもの<br>
    <table border='1'>%s</table>
    </body></html>""" % (ngstr, lowstr))
