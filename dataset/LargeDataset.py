from dataset.AbstractDataset import AbstractDataset
from tensorflow.python.ops import array_ops

import tensorflow as tf
import tensorflow.data as data

import config.baseConfig as baseConfig
import config.classes
import util.image

import numpy as np

def to_index(label):
    return config.classes[label]

def parse_csv(line):
    line = line.decode('utf-8')
    return line

def img2vector(path, config, Container):
    contents = tf.read_file(path)
    img = tf.image.decode_image(contents)
    img = tf.image.resize_image_with_crop_or_pad(img, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
    img = tf.scalar_mul(1/255.0, tf.cast(tf.reshape(img, [config.IMAGE_SIZE[0]* config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL]), baseConfig.floatSize))
    return Container.get("sess").run(img)


class LargeDataset(AbstractDataset):
    def __init__(self, csvpath, config, batch_size):
        def makeImage(img):
            img = tf.image.resize_image_with_crop_or_pad(img, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
            return tf.scalar_mul(1/255.0, tf.cast(tf.reshape(img, [config.IMAGE_SIZE[0]* config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL]), baseConfig.floatSize))

        def read_image(filename):
            contents = tf.read_file(filename)
            return makeImage(tf.image.decode_image(contents))

        def make_label(labelId):
            return tf.one_hot(labelId, config.NUM_CLASSES)

        self.batch_size = batch_size

        self._config = config
        file = open(csvpath, 'r')
        imgPaths = []
        labelIds = []
        n = 0
        for line in file:
            imgPath, labelId= line.split(",")
            imgPaths.append(imgPath)
            labelIds.append(int(labelId))
            n += 1
        self.length = n
            
        self._createDataset =  tf.data.Dataset.from_tensor_slices(imgPaths)\
            .map(read_image)\
            .batch(self.batch_size)
        self._readDataset =  tf.data.Dataset.from_tensor_slices(labelIds)\
            .map(make_label)\
            .batch(self.batch_size)
        self._iterator = self._createDataset.make_initializable_iterator()
        self._iterator2 = self._readDataset.make_initializable_iterator()
        self._next_elem = self._iterator.get_next()
        self._next_elem2 = self._iterator2.get_next()

    def getLen(self):
        return self.length

    def train(self, sess, op, phs, dropout=0.5):
        with tf.device("cpu:0"):
            sess.run(self._iterator.initializer)
            sess.run(self._iterator2.initializer)
        for i in range(10000):
            with tf.device("cpu:0"):
                try:
                    trains = sess.run(self._next_elem)
                    labels = sess.run(self._next_elem2)
                    if len(trains) == 0:
                        break
                except Exception as e:
                    break
            sess.run(op, feed_dict=phs.getDict(trains, labels, dropout, True))
  
    def calcAccuracy(self, sess, op, phs):
        acc_sum = 0
        with tf.device("cpu:0"):
            sess.run(self._iterator.initializer)
            sess.run(self._iterator2.initializer)
        for i in range(1000):
            with tf.device("cpu:0"):
                try:
                    trains = sess.run(self._next_elem)
                    labels = sess.run(self._next_elem2)
                    if len(trains) == 0:
                        break
                except Exception as e:
                    break
            acc_sum += sess.run(op, feed_dict=phs.getDict(trains, labels, 1.0))
        return acc_sum / self.length
