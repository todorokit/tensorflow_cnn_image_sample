import sys, re

import tensorflow as tf
import tensorflow.python.platform

import config

def makeVar(name, shape, initializer):
    # 1 gpu の場合、cpuに保存しない方が速い。multi gpu の場合, cpu側のメモリに保存しなくてはならない
    # リファクタリングして class化する時に1gpuの場合を自動で考慮する。
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

def inference(images_placeholder, imageSize, numInitChannel, conv2dList, numClasses,wscale, keep_prob, reuse_variables):
    def weight_variable(tuneArray, shape, wscale= 0.1):
#      print (shape, wscale)
      v = makeVar("W", shape, initializer=tf.truncated_normal_initializer(stddev=wscale))
      if (tuneArray is not None):
          tuneArray.append(v)
      return v

    def bias_variable(tuneArray, shape):
      v = makeVar("b", shape, initializer=tf.constant_initializer(0.1))
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
        with tf.variable_scope(name+"_var", reuse = reuse_variables) as vscope:
            with tf.name_scope(name) as scope:
                W = weight_variable(tuneArray, [filterSize, filterSize, inChannel, outChannel], wscale)
                b = bias_variable(tuneArray, [outChannel])
                return tf.nn.conv2d(x, W, strides=[1, slide, slide, 1], padding='SAME')+ b

    def linear(tuneArray, name, x, inChannel, outChannel):
        with tf.variable_scope(name+"_var", reuse = reuse_variables) as vscope:
            with tf.name_scope(name) as scope:
                W = weight_variable(tuneArray, [inChannel, outChannel], wscale)
                b = bias_variable(tuneArray, [outChannel])
                return tf.matmul(h, W) + b

    tuneArray = []
    prevChannel = numInitChannel
    h = tf.reshape(images_placeholder, [-1, imageSize, imageSize, numInitChannel])

    for name, filterSize, channel in conv2dList:
        if (re.search("^pool", name)):
            h = max_pool(name, h, filterSize)
            imageSize = imageSize // 2
        elif (re.search("^flatten", name)):
            prevChannel *= imageSize * imageSize
            h = tf.reshape(h, [-1,  prevChannel])
        elif (re.search("^dropout", name)):
            h = tf.nn.dropout(h, keep_prob)
        elif (re.search("^fc", name)):
            func = filterSize
            h = func(linear(None, name, h, prevChannel, channel))
            prevChannel = channel
        elif (re.search("^norm", name)):
            depth_radius = filterSize
            norm1 = tf.nn.lrn(h, depth_radius, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
        elif re.search("^conv", name):
            h = tf.nn.relu(conv2dWeightBias(tuneArray, name, h , prevChannel, channel, filterSize, 1, wscale))
            prevChannel = channel
        else:
            raise Exception("modelcnn::inference config.conv2dList no supported method")
            
    return (h, tuneArray)

def loss(logits, labels):
    # Mul
#    print(logits.shape)
#    print(labels.shape)
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
#    tf.summary.scalar("cross_entropy", cross_entropy)
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
def calcAccuracy(sess, batch_size, op, images_placeholder, labels_placeholder, keep_prob, acc_batch_size, images, labels):
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
            keep_prob: 1.0,
            batch_size: acc_batch_size
        })
    acc_batch = acc_batch_size*acc_loop
    if acc_batch != acc_n:
#        print(acc_batch, acc_n)
        acc_sum += sess.run(op, feed_dict={
            images_placeholder: images[acc_batch:acc_n],
            labels_placeholder: labels[acc_batch:acc_n],
            keep_prob: 1.0,
            batch_size: acc_batch_size
        })
    return acc_sum / acc_n

# model tutrials image cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

## FIXME: make class. not use FLAGS
def multiGpuLearning(FLAGS, imagesPh, labelsPh, keep_prob, batch_size, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, NUM_CLASSES, wscale):
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        logitsList = []
        lossList = []
        towerGrads = []
        tuneArrays = []
#        opt = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.9, beta2=0.999)
        opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

        reuseVar = False
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_id in range(0, FLAGS.num_gpu):
                print("GPU ID:" +str(gpu_id))
                in_batch_length = tf.div(batch_size, FLAGS.num_gpu)
                in_batch_start = tf.multiply(in_batch_length, gpu_id)
                slice_start = [in_batch_start, 0]
                slicedImagePh = tf.slice(imagesPh, slice_start, [in_batch_length, IMAGE_SIZE * IMAGE_SIZE* NUM_RGB_CHANNEL])
                slicedLabelPh = tf.slice(labelsPh, slice_start, [in_batch_length, NUM_CLASSES])
                with tf.device("/gpu:"+str(gpu_id)):
                    with tf.name_scope("tower_"+str(gpu_id)):
                        logits, tuneArray = inference(slicedImagePh, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, NUM_CLASSES, wscale, keep_prob, reuseVar)
                        # 1回目はinferenceでreuse(参照)しない。2回目以降はreuse(参照)する。
                        reuseVar = True
                        tuneArrays.append(tuneArray)
                        logitsList.append((logits, slicedLabelPh))
                        loss_value = loss(logits, slicedLabelPh)
                        lossList.append(loss_value)
                        grads = opt.compute_gradients(loss_value)
                        towerGrads.append(grads)

            grads = average_gradients(towerGrads)
            train_op = opt.apply_gradients(grads, global_step=global_step)

            # adam では殆どの計算をcpuで行なっている。
#            vscope = tf.get_variable_scope()
#            train_op = opt.minimize(tf.add_n(lossList))
            acc_op = tf.add_n([accuracy(logits, labels) for logits, labels in logitsList])
            return train_op, acc_op, tuneArrays
