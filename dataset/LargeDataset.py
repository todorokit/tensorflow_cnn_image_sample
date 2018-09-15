import tensorflow as tf

from dataset.AbstractDataset import AbstractDataset
import config.baseConfig as baseConfig

def img2vector(path, config, Container):
    contents = tf.read_file(path)
    img = tf.image.decode_image(contents)
    shape = tf.shape(img)
    height = tf.cond(shape[0] > shape[1], lambda : shape[1], lambda :shape[0])
    img = tf.image.resize_image_with_crop_or_pad(img, height, height)
    img = tf.image.resize_images(img, [config.IMAGE_SIZE[1],config.IMAGE_SIZE[0]])
    img = tf.scalar_mul(1/255.0, tf.cast(tf.reshape(img, [config.IMAGE_SIZE[0]* config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL]), baseConfig.floatSize))
    return Container.get("sess").run(img)

class LargeDataset(AbstractDataset):
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

        self._imageDataset =  tf.data.Dataset.from_tensor_slices(imgPaths)\
            .map(read_image)\
            .batch(self.batch_size)
        self._labelDataset =  tf.data.Dataset.from_tensor_slices(labelIds)\
            .map(make_label)\
            .batch(self.batch_size)
        if cache:
            self._imageDataset.cache()
            self._labelDataset.cache()
        self._iterator = self._imageDataset.make_initializable_iterator()
        self._iterator2 = self._labelDataset.make_initializable_iterator()
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
