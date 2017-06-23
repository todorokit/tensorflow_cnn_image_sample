import sys, re
from pprint import pprint

import tensorflow as tf
import tensorflow.python.platform

# fixme
import config

def inference(imagePh, keepProb, imageSize, numInitChannel, conv2dList, wscale, reuse = False, phaseTrain = None):
    tuneArray = []
    h = tf.reshape(imagePh, [-1, imageSize[0], imageSize[1], numInitChannel])

    for layer in conv2dList:
#        print(klass.name)
#        print(h.shape)
        h = layer.apply(tuneArray, h, wscale, phaseTrain, keepProb, reuse)
    return (h, tuneArray)

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
#    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))
    return accuracy

class Placeholders():
    def __init__(self, imageSize, numChannel, numClasses, is_training = False):
        self.imagesPh = tf.placeholder("float", shape=(None, imageSize[0]*imageSize[1]*numChannel))
        self.labelsPh = tf.placeholder("float", shape=(None, numClasses))
        self.keep_prob = tf.placeholder("float")
        self.batch_size = tf.placeholder("int32")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train') if is_training else None

    def getImages(self):
        return self.imagesPh

    def getLabels(self):
        return self.labelsPh

    def getKeepProb(self):
        return self.keep_prob

    def getBatchSize(self):
        return self.batch_size

    def getPhaseTrain(self):
        return self.phase_train

    def getDict(self, images, labels, keepProb, is_training = False):
        if self.getPhaseTrain() is None:
            return {
                self.getImages(): images,
                self.getLabels(): labels,
                self.getKeepProb(): keepProb,
                self.getBatchSize(): len(images),
            }
        else:
            return {
                self.getImages(): images,
                self.getLabels(): labels,
                self.getKeepProb(): keepProb,
                self.getBatchSize(): len(images),
                self.getPhaseTrain(): is_training
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
        batch = None
        for i in range(self.numAccuracyLoop):
            batch = self.acc_batch_size * i
            #print(batch, batch+self.acc_batch_size)
            self.splitedAccuracyImage.append(self.images[batch:batch+self.acc_batch_size])
            self.splitedAccuracyLabel.append(self.labels[batch:batch+self.acc_batch_size])

        if batch is None:
            batch = 0
        else:
            batch = batch + self.acc_batch_size
        if batch != n:
            self.numAccuracyLoop += 1
            #print(batch, n)
            self.splitedAccuracyImage.append(self.images[batch:n])
            self.splitedAccuracyLabel.append(self.labels[batch:n])

        batch = None
        n = len(self.testImages)
        self.numTestAccuracyLoop = n //self.acc_batch_size
        self.splitedTestAccuracyImage = []
        self.splitedTestAccuracyLabel = []
        for i in range(self.numTestAccuracyLoop):
            batch = self.acc_batch_size * i
            #print(batch, batch+self.acc_batch_size)
            self.splitedTestAccuracyImage.append(self.testImages[batch:batch+self.acc_batch_size])
            self.splitedTestAccuracyLabel.append(self.testLabels[batch:batch+self.acc_batch_size])
        if batch is None:
            batch = 0
        else:
            batch = batch + self.acc_batch_size
        if batch != n:
            self.numTestAccuracyLoop += 1
            #print(batch, n)
            self.splitedTestAccuracyImage.append(self.testImages[batch:n])
            self.splitedTestAccuracyLabel.append(self.testLabels[batch:n])

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

def calcAccuracy(sess, op, phs,dataset, isTest = False):
    acc_sum = 0
    for i in dataset.getAccuracyLoop(isTest):
        acc_sum += sess.run(op, feed_dict=phs.getDict(
            dataset.getAccuracyImage(i, isTest),
            dataset.getAccuracyLabel(i, isTest),
            1.0
        ))
    return acc_sum / dataset.getAccuracyLen(isTest)

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
def multiGpuLearning(learning_rate, phs, imageSize, numRGBChannel, conv2dList, numClasses, wscale):
    debug = tf.constant(1)
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        logitsList = []
        towerGrads = []
        xtuneArray = None
        opt = tf.train.AdamOptimizer(learning_rate)

        reuseVar = False
        with tf.variable_scope(tf.get_variable_scope()):
            # FIXME config.num_gpu
            for gpu_id in range(0, config.num_gpu):
                # print("GPU ID:" +str(gpu_id))
                in_batch_length = tf.div(phs.getBatchSize(), config.num_gpu)
                in_batch_start = tf.multiply(in_batch_length, gpu_id)
                slice_start = [in_batch_start, 0]
                slicedImagePh = tf.slice(phs.getImages(), slice_start, [in_batch_length, imageSize[0] * imageSize[1]* numRGBChannel])
                slicedLabelPh = tf.slice(phs.getLabels(), slice_start, [in_batch_length, numClasses])
                with tf.device("/gpu:"+str(gpu_id)):
                    with tf.name_scope("tower_"+str(gpu_id)):
                        logits, tuneArray = inference(slicedImagePh, phs.getKeepProb(), imageSize, numRGBChannel, conv2dList, wscale, reuseVar, phs.getPhaseTrain())
                        # 1回目はinferenceでreuse(参照)しない。
                        # 2回目以降はreuse(参照)する。
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
