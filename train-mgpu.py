#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time
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
flags.DEFINE_integer('num_gpu', 2, 'Number of GPU.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, 'Training batch size. This must divide evenly into the train dataset sizes.')
flags.DEFINE_integer('acc_batch_size', 1860, 'Accuracy batch size. This must divide evenly into the test data set sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_string('is_continue', "", 'Initial learning rate.')
flags.DEFINE_string('is_best', "", 'Initial learning rate.')

def getBest():
    path = os.path.join("best-model", "score-multi.txt")
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
        save_path = saver.save(sess, os.path.join(cwd, config.modelMultiFile))
        deeptool.backup(config.modelMultiFile, "best-model")
        path = os.path.join("best-model", "score-multi.txt")
        fp = open(path, "w")
        fp.write(str(score))
        fp.close()

if __name__ == '__main__':
    train_image, train_label, _ = deeptool.loadImages(FLAGS.train, IMAGE_SIZE, NUM_CLASSES)
    test_image, test_label, _ =  deeptool.loadImages(FLAGS.test, IMAGE_SIZE, NUM_CLASSES)
    cwd = os.getcwd()
    with tf.Graph().as_default():
        imagesPh = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL))
        labelsPh = tf.placeholder("float", shape=(None, NUM_CLASSES))
        keep_prob = tf.placeholder("float")
        batch_size = tf.placeholder("int32")
        train_op, acc_op, tuneArrays, debug = modelcnn.multiGpuLearning(FLAGS, imagesPh, labelsPh, keep_prob, batch_size, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, NUM_CLASSES, WSCALE)

        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())
#        if FLAGS.is_continue != "":
#            saver.restore(sess, os.path.join(cwd, config.modelMultiFile))

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # 訓練の実行
        n = int(len(train_image)/FLAGS.batch_size)
        for step in range(FLAGS.max_steps):
            startTime = time.time()
            for i in range(n):
                batch = FLAGS.batch_size*i
                sess.run(train_op, feed_dict={
                    imagesPh: train_image[batch:batch+FLAGS.batch_size],
                    labelsPh: train_label[batch:batch+FLAGS.batch_size],
                    keep_prob: 0.5,
                    batch_size: FLAGS.batch_size})

            train_accuracy = modelcnn.calcAccuracy(sess, batch_size, acc_op, imagesPh, labelsPh, keep_prob, FLAGS.acc_batch_size, train_image, train_label)
            test_accuracy = modelcnn.calcAccuracy(sess, batch_size, acc_op, imagesPh, labelsPh, keep_prob, FLAGS.acc_batch_size, test_image, test_label)
            writeBest(sess,saver,test_accuracy)

            print("step %d, training accuracy %g, test accuracy %g, %d batch/sec"%(step, train_accuracy, test_accuracy, int(n/(time.time() - startTime))))
            sys.stdout.flush()
#            summary_str = sess.run(summary_op, feed_dict=feedDictNoProb)
#            summary_writer.add_summary(summary_str, step)

    save_path = saver.save(sess, os.path.join(cwd, config.modelMultiFile))
