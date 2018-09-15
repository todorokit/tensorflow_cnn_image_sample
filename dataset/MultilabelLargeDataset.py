import tensorflow as tf
import numpy as np

from dataset.AbstractDataset import AbstractDataset
import config.baseConfig as baseConfig

class MultilabelLargeDataset(AbstractDataset):
    def __init__(self, csvpath, config, batch_size, cache):
        def makeImage(img):
            shape = tf.shape(img)
            height = tf.cond(shape[0] > shape[1], lambda : shape[1], lambda :shape[0])
            img = tf.image.resize_image_with_crop_or_pad(img, height, height)
            img = tf.image.resize_images(img, [config.IMAGE_SIZE[1],config.IMAGE_SIZE[0]])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=63)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            return tf.scalar_mul(1/255.0, tf.cast(tf.reshape(img, [config.IMAGE_SIZE[0]* config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL]), baseConfig.floatSize))

        def read_image(filename):
            contents = tf.read_file(filename)
            return makeImage(tf.image.decode_image(contents))

        self.batch_size = batch_size

        self._config = config
        with tf.device("cpu:0"):
            file = open(csvpath, 'r')
            imgPaths = []
            labelIds = []
            labelBatchs = []
            n = 0
            for line in file:
                imgPath, *labels= line.split(",")
                imgPaths.append(imgPath)
                # 出力は float 32bit (softmax が32bitだから)
                labelIds.append(np.array(labels, dtype=np.float32))
                if len(labelIds) == batch_size:
                    labelBatchs.append(labelIds)
                    labelIds = []
                n += 1
            if len(labelIds) > 0:
                labelBatchs.append(labelIds)
            self.labelBatchs = labelBatchs
            self.length = n
            
            self._imageDataset =  tf.data.Dataset.from_tensor_slices(imgPaths)\
                                                  .map(read_image)\
                                                  .batch(self.batch_size)
            if cache:
                self._imageDataset.cache()

            self._iterator = self._imageDataset.make_initializable_iterator()
            self._next_elem = self._iterator.get_next()

    def getLen(self):
        return self.length

    def train(self, sess, op, phs, dropout=0.5):
        with tf.device("cpu:0"):
            sess.run(self._iterator.initializer)
        for labels in self.labelBatchs:
            with tf.device("cpu:0"):
                trains = sess.run(self._next_elem)
            sess.run(op, feed_dict=phs.getDict(trains, labels, dropout, True))
  
    def calcAccuracy(self, sess, op, phs):
        acc_sum = 0
        with tf.device("cpu:0"):
            sess.run(self._iterator.initializer)
        for labels in self.labelBatchs:
            with tf.device("cpu:0"):
                trains = sess.run(self._next_elem)
            acc_sum += sess.run(op, feed_dict=phs.getDict(trains, labels, 1.0))
        return acc_sum / self.length
