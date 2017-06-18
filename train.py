#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
from pprint import pprint

import tensorflow as tf
import tensorflow.python.platform

import config, deeptool, modelcnn


NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
conv2dList=config.conv2dList
WSCALE=config.WSCALE

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', 'c:\\tmp\\image_cnn', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, 'Training batch size. This must divide evenly into the train dataset sizes.')
flags.DEFINE_integer('acc_batch_size', 1860, 'Accuracy batch size. Take care of memory limit.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_string('is_continue', "", 'Initial learning rate.')
flags.DEFINE_string('is_best', "", 'Initial learning rate.')

def getBest():
    path = os.path.join("best-model", "score.txt")
    if os.path.exists(path):
        fp = open(path, "r")
        score = fp.read().replace("\n", "")
        fp.close()
        return float(score)
    else:
        return 0.0

def writeBest(sess, saver, score):
    if (FLAGS.is_best != "" and getBest() < score):
        cwd = os.getcwd()
        save_path = saver.save(sess, os.path.join(cwd, config.modelFile))
        deeptool.backup(config.modelFile, "best-model")
        path = os.path.join("best-model", "score.txt")
        fp = open(path, "w")
        fp.write(str(score))
        fp.close()

if __name__ == '__main__':
    train_image, train_label, _ = deeptool.loadImages(FLAGS.train, IMAGE_SIZE, NUM_CLASSES)
    test_image, test_label, _ =  deeptool.loadImages(FLAGS.test, IMAGE_SIZE, NUM_CLASSES)
    
    cwd = os.getcwd()
    with tf.Graph().as_default():
        phs = modelcnn.Placeholders(IMAGE_SIZE, NUM_RGB_CHANNEL, NUM_CLASSES, True)
        dataset = modelcnn.InMemoryDataset(train_image, train_label, test_image, test_label, FLAGS.batch_size, FLAGS.acc_batch_size)
        logits, _ = modelcnn.inference(phs.getImages(), phs.getKeepProb(), IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, WSCALE, False, phs.getPhaseTrain())
        loss_value = modelcnn.loss(logits, phs.getLabels())
        train_op = modelcnn.training(loss_value, FLAGS.learning_rate)
        acc_op = modelcnn.accuracy(logits, phs.getLabels())

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if FLAGS.is_continue != "":
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(cwd, config.modelFile))

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        n = len(train_image)
        for step in range(FLAGS.max_steps):
            startTime = time.time()
            for i in dataset.getTrainLoop():
                sess.run(train_op, feed_dict=phs.getDict(
                    dataset.getTrainImage(i),
                    dataset.getTrainLabel(i),
                    0.5,
                    True
                ))

            train_accuracy = modelcnn.calcAccuracy(sess, acc_op, phs, dataset)
            test_accuracy = modelcnn.calcAccuracy(sess, acc_op, phs, dataset, isTest = True)
            writeBest(sess,saver,test_accuracy)

            print("step %d, training accuracy %g, test accuracy %g, %g data/sec"%(step, train_accuracy, test_accuracy, n/(time.time() - startTime)))
            sys.stdout.flush()
#            summary_str = sess.run(summary_op, feed_dict=feedDictNoProb)
#            summary_writer.add_summary(summary_str, step)

    save_path = saver.save(sess, os.path.join(cwd, config.modelFile))
