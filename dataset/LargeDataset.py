import tensorflow as tf

from dataset.AbstractDataset import AbstractDataset
import config.baseConfig as baseConfig

def img2vector(path, config):
    contents = tf.read_file(path)
    img = tf.image.decode_image(contents)
    shape = tf.shape(img)
    height = tf.cond(shape[0] > shape[1], lambda : shape[1], lambda :shape[0])
    img = tf.image.resize_image_with_crop_or_pad(img, height, height)
    img = tf.image.resize_images(img, [config.IMAGE_SIZE[1],config.IMAGE_SIZE[0]])
    imgop = tf.scalar_mul(1/255.0, tf.cast(tf.reshape(img, [config.IMAGE_SIZE[0]* config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL]), baseConfig.floatSize))
    return imgop

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
             .map(read_image)
        self._labelDataset =  tf.data.Dataset.from_tensor_slices(labelIds)\
            .map(make_label)
        
        self._dataset = tf.data.Dataset.zip((self._imageDataset, self._labelDataset))\
            .shuffle(self.batch_size)\
            .batch(self.batch_size)
        
        if cache :
            self._dataset.cache()
        self._iterator = self._dataset.make_initializable_iterator()
        self._next_elem = self._iterator.get_next()

    def getLen(self):
        return self.length

    def train(self, sess, op, acc_op, phs, saver, dropout=0.5):
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
            sess.run(op, feed_dict=phs.getDict(trains, labels, dropout, True))
            if loop % 100 == 99:
                acc = sess.run(acc_op, feed_dict=phs.getDict(trains, labels, 1.0)) / len(trains)
                saver.save("train-loss: %g"% acc)
  
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
