#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import os
import config
import time

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = 3
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL

manip = 32
FILTER1_SIZE = 5
IMAGE_SIZE_1 = IMAGE_SIZE  // 2

manip2 = 64
FILTER2_SIZE = 5
IMAGE_SIZE_2 = IMAGE_SIZE_1 // 2

fc1_manip = 1024

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', 'c:\\tmp\\image_cnn', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 30, 'Number of steps to run trainer.')
# batch_size は計算速度だけではなく、収束にも影響あり
flags.DEFINE_integer('batch_size', 20, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

def inference(images_placeholder, keep_prob):
    """ 予測モデル keep_probはトレーニング時以外は1.0にする    """
    
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを0.1で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_RGB_CHANNEL])
    # output shape [batch_size  , IMAGE_WIDTH, IMAGE_HEIGHT, NUM_RGB_CHANNEL]

    # 畳み込み層1
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([FILTER1_SIZE, FILTER1_SIZE, NUM_RGB_CHANNEL, manip])
        b_conv1 = bias_variable([manip])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # output shape [batch_size  , IMAGE_WIDTH, IMAGE_HEIGHT, manip]

    # プーリング層1
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    # output shape [batch_size  , IMAGE_WIDTH//2, IMAGE_HEIGHT//2, manip]
    
    # 畳み込み層2
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([FILTER2_SIZE, FILTER2_SIZE, manip, manip2])
        b_conv2 = bias_variable([manip2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # output shape [batch_size  , IMAGE_WIDTH, IMAGE_HEIGHT, manip2]

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    # output shape [batch_size  , IMAGE_WIDTH//4, IMAGE_HEIGHT//4, manip2]

    # 全結合層1
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([IMAGE_SIZE_2*IMAGE_SIZE_2*manip2, fc1_manip])
        b_fc1 = bias_variable([fc1_manip])
        h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_SIZE_2*IMAGE_SIZE_2*manip2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # output shape [batch_size  , fc1_manip]

    # 全結合層2
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([fc1_manip, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # output shape [batch_size, NUM_CLASSES]

    return y_conv

def loss(logits, labels):
    """ 誤差関数: 最適化の評価用に交差エントロピーの計算をする """
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.summary.scalar("cross_entropy", cross_entropy) # for tensorboard
    return cross_entropy

def training(loss, learning_rate):
    """ トレーニング: モデルの変数をlossを目安に最適化する """
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数 """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy) # for tensorboard
    return accuracy

def loadImages(labelFilePath):
    file = open(labelFilePath, 'r')
    image = []
    label = []
    for line in file:
        imgpath, labelIndex= line.rstrip().split()
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        image.append(img.flatten().astype(np.float32)/255.0)
        labelData = np.zeros(NUM_CLASSES)
        labelData[int(labelIndex)] = 1
        label.append(labelData)
    file.close()
    return (np.asarray(image), np.asarray(label))

if __name__ == '__main__':
    train_image, train_label = loadImages(FLAGS.train)
    test_image, test_label =  loadImages(FLAGS.test)
    
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        keep_prob = tf.placeholder("float")

        logits = inference(images_placeholder, keep_prob)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, FLAGS.learning_rate)

        acc = accuracy(logits, labels_placeholder)

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

        print("test accuracy %g"%sess.run(acc, feed_dict={
            images_placeholder: test_image,
            labels_placeholder: test_label,
            keep_prob: 1.0}))

    cwd = os.getcwd()
    save_path = saver.save(sess, cwd+"\\model.ckpt")
