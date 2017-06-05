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

def inference(imagePh, keepProb, imageSize, numInitChannel, conv2dList, numClasses, wscale, reuse_variables = False):
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
    h = tf.reshape(imagePh, [-1, imageSize, imageSize, numInitChannel])

    for name, filterSize, channel in conv2dList:
        if (re.search("^pool", name)):
            h = max_pool(name, h, filterSize)
            imageSize = imageSize // 2
        elif (re.search("^flatten", name)):
            prevChannel *= imageSize * imageSize
            h = tf.reshape(h, [-1,  prevChannel])
        elif (re.search("^dropout", name)):
            h = tf.nn.dropout(h, keepProb)
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

class Placeholders():
    def __init__(self, imageSizeW, imageSizeH, numChannel, numClasses):
        self.imagesPh = tf.placeholder("float", shape=(None, imageSizeW*imageSizeH*numChannel))
        self.labelsPh = tf.placeholder("float", shape=(None, numClasses))
        self.keep_prob = tf.placeholder("float")
        self.batch_size = tf.placeholder("int32")

    def getImages(self):
        return self.imagesPh

    def getLabels(self):
        return self.labelsPh

    def getKeepProb(self):
        return self.keep_prob

    def getBatchSize(self):
        return self.batch_size

    def getDict(self, images, labels, keepProb):
        return {
            self.getImages(): images,
            self.getLabels(): labels,
            self.getKeepProb(): keepProb,
            self.getBatchSize(): len(images)
        }
    
## in Memory Dataset
class InMemoryDataset():
    def __init__(self, images, labels, testImages, testLabels, batch_size, acc_batch_size):
        self.images = images
        self.labels = labels
        self.testImages = testImages
        self.testLabels = testLabels
        self.batch_size = batch_size
        self.acc_batch_size = acc_batch_size
        self.splitedImage = None
        self.splitedLabel = None
        self.splitedAccracyImage = None
        self.splitedAccracyLabel = None
        self.splitedTestAccracyImage = None
        self.splitedTestAccracyLabel = None
        self.numTrainLoop = 0
        self.numAccuracyLoop = 0
        self.numTestAccuracyLoop = 0
        self.calcBatchImage()
        
    def calcBatchImage(self):
        n = len(self.images)
        self.numTrainLoop = n // self.batch_size
        self.splitedImage = []
        self.splitedLabel = []
        # train は割り切って下さい
        for i in range(self.numTrainLoop):
            batch = self.batch_size * i
            self.splitedImage.append(self.images[batch:batch+self.batch_size])
            self.splitedLabel.append(self.labels[batch:batch+self.batch_size])

        self.numAccuracyLoop = n//self.acc_batch_size
        self.splitedAccuracyImage = []
        self.splitedAccuracyLabel = []
        for i in range(self.numAccuracyLoop):
            batch = self.acc_batch_size * i
            #print(batch, batch+self.acc_batch_size)
            self.splitedAccuracyImage.append(self.images[batch:batch+self.acc_batch_size])
            self.splitedAccuracyLabel.append(self.labels[batch:batch+self.acc_batch_size])

        batch = batch + self.acc_batch_size
        if batch != n:
            self.numAccuracyLoop += 1
            #print(batch, n)
            self.splitedAccuracyImage.append(self.images[batch:n])
            self.splitedAccuracyLabel.append(self.labels[batch:n])

        n = len(self.testImages)
        self.numTestAccuracyLoop = n //self.acc_batch_size
        self.splitedTestAccuracyImage = []
        self.splitedTestAccuracyLabel = []
        for i in range(self.numTestAccuracyLoop):
            batch = self.acc_batch_size * i
            #print(batch, batch+self.acc_batch_size)
            self.splitedTestAccuracyImage.append(self.testImages[batch:batch+self.acc_batch_size])
            self.splitedTestAccuracyLabel.append(self.testLabels[batch:batch+self.acc_batch_size])
        batch = batch + self.acc_batch_size
        if batch != n:
            self.numTestAccuracyLoop += 1
            #print(batch, n)
            self.splitedTestAccuracyImage.append(self.images[batch:n])
            self.splitedTestAccuracyLabel.append(self.labels[batch:n])

    def getAccBatchSize(self):
        return self.acc_batch_size

    def getTrainLoop(self):
        return range(self.numTrainLoop)
    def getTrainImage(self, i):
        return self.splitedImage[i]
    def getTrainLabel(self, i):
        return self.splitedLabel[i]

    def getAccuracyLoop(self, isTest= False):
        if isTest:
            return range(self.numTestAccuracyLoop)
        else:
            return range(self.numAccuracyLoop)
    def getAccuracyImage(self, i, isTest = False):
        if isTest:
            return self.splitedTestAccuracyImage[i]
        else:
            return self.splitedAccuracyImage[i]
    def getAccuracyLabel(self, i, isTest= False):
        if isTest:
            return self.splitedTestAccuracyLabel[i]
        else:
            return self.splitedAccuracyLabel[i]

    def getAccuracyLen(self, isTest= False):
        if isTest:
            return len(self.testImages)
        else:
            return len(self.images)

    
# all data version
def calcAccuracy(sess, op, phs,dataset, isTest = False):
    acc_sum = 0
    for i in dataset.getAccuracyLoop(isTest):
        acc_sum += sess.run(op, feed_dict=phs.getDict(
            dataset.getAccuracyImage(i, isTest),
            dataset.getAccuracyLabel(i, isTest),
            1.0
        ))
    return acc_sum / dataset.getAccuracyLen(isTest)

# model tutrials image cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  average_grads = []
  # 各variableに対して
  for grad_and_vars in zip(*tower_grads):
    grads = []

    # 各GPUの結果に対して 平均を取る
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    # v は 全GPU同じはず
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

## FIXME: make parameter class. not use FLAGS
def multiGpuLearning(learning_rate, phs, IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, NUM_CLASSES, wscale):
    debug = tf.constant(1)
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        logitsList = []
        towerGrads = []
        xtuneArray = None
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

        reuseVar = False
        with tf.variable_scope(tf.get_variable_scope()):
            # FIXME config.num_gpu
            for gpu_id in range(0, config.num_gpu):
                # print("GPU ID:" +str(gpu_id))
                # data must batch_size == len(phs.getImages())
                in_batch_length = tf.div(phs.getBatchSize(), config.num_gpu)
                in_batch_start = tf.multiply(in_batch_length, gpu_id)
                slice_start = [in_batch_start, 0]
                slicedImagePh = tf.slice(phs.getImages(), slice_start, [in_batch_length, IMAGE_SIZE * IMAGE_SIZE* NUM_RGB_CHANNEL])
                slicedLabelPh = tf.slice(phs.getLabels(), slice_start, [in_batch_length, NUM_CLASSES])
                with tf.device("/gpu:"+str(gpu_id)):
                    with tf.name_scope("tower_"+str(gpu_id)):
                        logits, tuneArray = inference(slicedImagePh, phs.getKeepProb(), IMAGE_SIZE, NUM_RGB_CHANNEL, conv2dList, NUM_CLASSES, wscale, reuseVar)
                        # 1回目はinferenceでreuse(参照)しない。
                        # 2回目以降はreuse(参照)する。
                        # Adam(scope外) は reuse = Falseが必要。
                        reuseVar = True
                        if xtuneArray is None:
                            xtuneArray = tuneArray
                        logitsList.append((logits, slicedLabelPh))
                        loss_value = loss(logits, slicedLabelPh)
                        grads = opt.compute_gradients(loss_value)
                        towerGrads.append(grads)

            grads = average_gradients(towerGrads)
            train_op = opt.apply_gradients(grads, global_step=global_step)

            acc_op = tf.add_n([accuracy(logits, labels) for logits, labels in logitsList])
            return train_op, acc_op, xtuneArray, debug
