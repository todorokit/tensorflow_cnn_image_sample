#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import tensorflow.python.platform

import config
import deeptool
import modelcnn

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', 'c:\\tmp\\image_cnn', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 50, 'Number of steps to run trainer.')
# batch_size は計算速度だけではなく、収束にも影響あり
flags.DEFINE_integer('batch_size', 10, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

if __name__ == '__main__':
    train_image, train_label, _ = deeptool.loadImages(FLAGS.train, IMAGE_SIZE, NUM_CLASSES)
    test_image, test_label, _ =  deeptool.loadImages(FLAGS.test, IMAGE_SIZE, NUM_CLASSES)
    
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL))
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        keep_prob = tf.placeholder("float")

        logits = modelcnn.inference(images_placeholder, keep_prob)
        loss_value = modelcnn.loss(logits, labels_placeholder)
        train_op = modelcnn.training(loss_value, FLAGS.learning_rate)

        acc = modelcnn.accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        feedDictNoProb = {
            images_placeholder: train_image,
            labels_placeholder: train_label,
            keep_prob: 1.0}
        
        # 訓練の実行
        n = int(len(train_image)/FLAGS.batch_size)
        for step in range(FLAGS.max_steps):
            startTime = time.time()
            for i in range(n):
                batch = FLAGS.batch_size*i
                sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                    labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                    keep_prob: 0.5})

            train_accuracy = sess.run(acc, feed_dict=feedDictNoProb)
            print("step %d, training accuracy %g, %d batch/sec"%(step, train_accuracy , int(n/(time.time() - startTime))))

            summary_str = sess.run(summary_op, feed_dict=feedDictNoProb)
            summary_writer.add_summary(summary_str, step)

        acct =sess.run(acc, feed_dict={
            images_placeholder: test_image,
            labels_placeholder: test_label,
            keep_prob: 1.0})
        f = open( "out.txt", "a")
        f.write("%g\n" %(acct))
        f.close()
        print("%g\n" %(acct))

    cwd = os.getcwd()
    save_path = saver.save(sess, cwd+"\\model.ckpt")
