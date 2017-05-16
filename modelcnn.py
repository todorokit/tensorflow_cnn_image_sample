import tensorflow as tf
import tensorflow.python.platform
import re

import config
           
def inference(images_placeholder, imageSize, numInitChannel, conv2dList, fc1Channel, numClasses,wscale, keep_prob):
    def weight_variable(tuneArray, shape, wscale= 0.1):
#      print (shape)
      initial = tf.truncated_normal(shape, stddev=wscale)
      v = tf.Variable(initial, name="W")
      if (tuneArray is not None):
          tuneArray.append(v)
      return v

    def bias_variable(tuneArray, shape):
      initial = tf.constant(0.1, shape=shape)
      v= tf.Variable(initial, name="b")
      if (tuneArray is not None):
          tuneArray.append(v)
      return v

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x, size = 2, slide = 2):
      return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                            strides=[1, slide, slide, 1], padding='SAME')
    
    def max_pool(name, x, size = 2):
        with tf.name_scope(name) as scope:
            return max_pool_2x2(x, size)


    def conv2dWeightBias(tuneArray, name, x, inChannel, outChannel, filterSize, slide = 1, wscale=1):
        with tf.name_scope(name) as scope:
            W = weight_variable(tuneArray, [filterSize, filterSize, inChannel, outChannel], wscale)
            b = bias_variable(tuneArray, [outChannel])
            return tf.nn.conv2d(x, W, strides=[1, slide, slide, 1], padding='SAME')+ b

    def linear(tuneArray, name, x, inChannel, outChannel):
        with tf.name_scope(name) as scope:
            W = weight_variable(tuneArray, [inChannel, outChannel])
            b = bias_variable(tuneArray, [outChannel])
            return tf.matmul(h, W) + b

    tuneArray = []
    prevChannel = numInitChannel
    h = tf.reshape(images_placeholder, [-1, imageSize, imageSize, numInitChannel])

    for name, filterSize, channel in conv2dList:
        if (re.search("^pool", name)):
            h = max_pool(name, h, filterSize)
            imageSize = imageSize // 2
        else:
            h = tf.nn.relu(conv2dWeightBias(tuneArray, name,h , prevChannel, channel, filterSize, 1, wscale))
            h = tf.nn.relu(conv2dWeightBias(tuneArray, name + "-a", h, channel, channel, 1, 1, wscale))
            h = tf.nn.relu(conv2dWeightBias(tuneArray, name +"-b", h, channel, channel, 1, 1, wscale))
            prevChannel = channel
    
    prevChannel *= imageSize * imageSize
    h = tf.reshape(h, [-1,  prevChannel])
#    h = tf.nn.relu(linear(tuneArray, name, h, prevChannel, fc1Channel))
#    prevChannel = fc1Channel
    h = tf.nn.dropout(h, keep_prob)
    # tuneArray (fine tuning) は最終層を持たない。
    y = tf.nn.softmax(linear(None, "fc2", h, prevChannel, numClasses))

    return (y, tuneArray)

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

# batch_size version
def accuracy(logits, labels):
    # logits.shape = labels.shape = [batch_size , num_class]
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))
    return accuracy

# all data version
def calcAccuracy(sess, op, images_placeholder, labels_placeholder, keep_prob, acc_batch_size, images, labels):
    acc_n = len(images)
    acc_loop = int(acc_n/acc_batch_size)
    acc_sum = 0
    
    acc_batch = 0
    for i in range(acc_loop):
        acc_batch = acc_batch_size*i
#        print(acc_batch, acc_batch+acc_batch_size)
        acc_sum += sess.run(op, feed_dict={
            images_placeholder: images[acc_batch:acc_batch+acc_batch_size],
            labels_placeholder: labels[acc_batch:acc_batch+acc_batch_size],
            keep_prob: 1.0})
    acc_batch = acc_batch_size*acc_loop
    if acc_batch != acc_n:
#        print(acc_batch, acc_n)
        acc_sum += sess.run(op, feed_dict={
            images_placeholder: images[acc_batch:acc_n],
            labels_placeholder: labels[acc_batch:acc_n],
            keep_prob: 1.0})
    return acc_sum / acc_n

