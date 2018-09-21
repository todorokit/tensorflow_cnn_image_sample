#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from util.Container import getContainer
from util.utils import *

flags = tf.app.flags
flags.DEFINE_string('config', "config.celeba", 'config module(file) name (no extension).')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
FLAGS = flags.FLAGS

def main(argv):
    with tf.Graph().as_default() as graph:

        Container = getContainer(FLAGS)
        config = Container.get("config")
        if config.num_gpu > 1 :
            gpumode = "MULTI GPU MODE"
            train_op, acc_op,_ ,_ ,phs = Container.get("ops_mgpu")
        else:
            gpumode = "SINGLE GPU MODE"
            train_op, acc_op,_ ,_, phs = Container.get("ops")

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, config.modelFile)  # 学習済みのグラフを読み込み

            #print(graph.as_graph_def())
            graph = graph.as_graph_def()
            tf.train.write_graph(graph, '.','/tmp/graph.pb', as_text=False)

if __name__ == '__main__':
    tf.app.run()
            
