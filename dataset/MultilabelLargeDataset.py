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

        def read_label(labels):
            return tf.string_to_number(tf.string_split([labels], delimiter=',').values, out_type=tf.float32)

        self.batch_size = batch_size

        self._config = config
        with tf.device("cpu:0"):
            file = open(csvpath, 'r')
            imgPaths = []
            labelIds = []
            n = 0
            for line in file:
                imgPath, labels= line.split(",", 1)
                imgPaths.append(imgPath)
                # 出力は float 32bit (softmax が32bitだから)
                labelIds.append(labels.strip())
                n += 1
            self.length = n
            
            self._imageDataset =  tf.data.Dataset.from_tensor_slices(imgPaths)\
                .map(read_image)
            self._labelDataset =  tf.data.Dataset.from_tensor_slices(labelIds)\
                .map(read_label)
        
            self._dataset = tf.data.Dataset.zip((self._imageDataset, self._labelDataset))\
                .shuffle(self.batch_size)\
                .batch(self.batch_size)
            if cache:
                self._dataset.cache()

            self._iterator = self._dataset.make_initializable_iterator()
            self._next_elem = self._iterator.get_next()
            

    def getLen(self):
        return self.length

    def train(self, sess, op, loss_op, extra_update_ops, phs, saver, mytimer,dropout=0.5):
        with tf.device("cpu:0"):
            sess.run(self._iterator.initializer)
        for loop in range(1000000):
            with tf.device("cpu:0"):
                try:
                    (trains, labels) = sess.run(self._next_elem)
                    if len(trains) == 0:
                        break
                except Exception as e:
                    break
            sess.run([op, extra_update_ops], feed_dict=phs.getDict(trains, labels, dropout, True))
            if loop % 500 == 499:
                acc = sess.run(loss_op, feed_dict=phs.getDict(trains, labels, 1.0)) / len(trains)
                saver.save("%s train-loss: %g"% (mytimer.getNow("%H:%M:%S"), acc))
  
    def calcAccuracy(self, sess, op, phs):
        acc_sum = 0
        with tf.device("cpu:0"):
            sess.run(self._iterator.initializer)
        for loop in range(1000000):
            with tf.device("cpu:0"):
                try:
                    (trains, labels) = sess.run(self._next_elem)
                    if len(trains) == 0:
                        break
                except Exception as e:
                    break
            acc_sum += sess.run(op, feed_dict=phs.getDict(trains, labels, 1.0))
        return acc_sum / self.length
