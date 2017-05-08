#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
from collections import OrderedDict

import tensorflow as tf
import tensorflow.python.platform

import config
import modelcnn
import deeptool

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('train_dir', 'c:\\tmp\\image_cnn', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 3, 'Number of steps to run trainer.')
flags.DEFINE_string('batch_sizes', "5,10", 'Batch size'
                    'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate_base', 2e-4, 'Initial learning rate.')
flags.DEFINE_float('learning_rate_odds', 0.95, 'Initial learning rate.')
flags.DEFINE_float('num_loop', 20, 'Initial learning rate.')

if __name__ == '__main__':
    train_image, train_label, _ = deeptool.loadImages(FLAGS.train, IMAGE_SIZE, NUM_CLASSES)
#    test_image, test_label =  deeptool.loadImages(FLAGS.test, IMAGE_SIZE, NUM_CLASSES)
    batch_sizes = [ int(sizestr) for sizestr in FLAGS.batch_sizes.split(",")]
    result = OrderedDict()

    for loop, batch_size in [(loop, batch_size) for loop in range(FLAGS.num_loop) for batch_size in batch_sizes]:
        
        with tf.Graph().as_default(), tf.Session() as sess:
            timeStart = time.time()
            images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_RGB_CHANNEL))
            labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
            keep_prob = tf.placeholder("float")
            
            logits = modelcnn.inference(images_placeholder, keep_prob)
            loss_value = modelcnn.loss(logits, labels_placeholder)
            acc = modelcnn.accuracy(logits, labels_placeholder)

            learningRate = FLAGS.learning_rate_base * (FLAGS.learning_rate_odds** loop)
            train_op = modelcnn.training(loss_value, learningRate)

            sess.run(tf.global_variables_initializer())
            
            feedDictNoProb = {
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0}
            
            # 訓練の実行
            n = int(len(train_image)/batch_size)
            for step in range(FLAGS.max_steps):
                for i in range(n):
                    batch = batch_size*i
                    sess.run(train_op, feed_dict={
                        images_placeholder: train_image[batch:batch+batch_size],
                        labels_placeholder: train_label[batch:batch+batch_size],
                        keep_prob: 0.50})
                
            timeTrain = time.time()
            accVal = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})

            # varbose = "(Time train %.2f, test %.2f)" % (timeTrain - timeStart, time.time() - timeTrain)
            fom = "test learningRate %.3e, batch_size %d, accuracy %g"
            print(fom%(learningRate, batch_size, accVal))
            sys.stdout.flush()
            i = "%.3e"%(learningRate)
            j = str(batch_size)
            try:
                result[i][j] = accVal
            except:
                dic = OrderedDict()
                dic[j] = accVal
                result[i] = dic
            #sess.close()
    print("---------------result-----------------")
    ## TODO: writeTsv() 
    sys.stdout.write("lr\t")
    for batch_size in batch_sizes:
        sys.stdout.write("%d\t"%(batch_size))
    sys.stdout.write("\n")
    for i in result:
        sys.stdout.write("%s\t"%(i))
        for j in result[i]:
            sys.stdout.write("%g\t"%(result[i][j]))
        sys.stdout.write("\n")
