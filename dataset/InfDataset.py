import tensorflow as tf

from dataset.AbstractDataset import AbstractDataset
import config.baseConfig as baseConfig

class InfDataset(AbstractDataset):
    def __init__(self, files, config, batch_size):
        def makeImage(img):
            shape = tf.shape(img)
            height = tf.cond(shape[0] > shape[1], lambda : shape[1], lambda :shape[0])
            img = tf.image.resize_image_with_crop_or_pad(img, height, height)
            img = tf.image.resize_images(img, [config.IMAGE_SIZE[1],config.IMAGE_SIZE[0]])
            if baseConfig.dataFormat == "channels_first":
                img = tf.transpose(img, perm=[0, 3, 1, 2])
            return tf.scalar_mul(1/255.0, tf.cast(tf.reshape(img, [config.IMAGE_SIZE[0]* config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL]), baseConfig.floatSize))

        def read_image(filename):
            contents = tf.read_file(filename)
            return makeImage(tf.image.decode_image(contents))

        self.batch_size = batch_size
        self._config = config
        self.length = len(files)
        
        self._imageDataset =  tf.data.Dataset.from_tensor_slices(files)\
            .map(read_image)
        self._labelDataset =  tf.data.Dataset.from_tensor_slices(files)

        self._dataset = tf.data.Dataset.zip((self._imageDataset, self._labelDataset))\
            .batch(self.batch_size)

        self._iterator = self._dataset.make_initializable_iterator()
        self._next_elem = self._iterator.get_next()

    def getLen(self):
        return self.length

    def inferenceDo(self, sess, op, phs, proc):
        acc_sum = 0
        ret = []
        with tf.device("cpu:0"):
            sess.run(self._iterator.initializer)
        for loop in range(1000000):
            with tf.device("cpu:0"):
                try:
                    (data, filenames) = sess.run(self._next_elem)
                    if len(data) == 0:
                        break
                except Exception as e:
                    break
            logits = sess.run(op, feed_dict={phs.getImages(): data, phs.getPhaseTrain():False, phs.getKeepProb(): 1.0})[0]
            top5s = map(top5, logits)
            for indice, filename, l in zip(top5s, filenames, logits):
                proc(indice, l, filename)
