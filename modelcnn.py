import tensorflow as tf
import tensorflow.python.platform

from config import baseConfig

flags = tf.app.flags
flags.DEFINE_string('config', "config.celeba", 'config module(file) name (no extension).')

def inference(imagePh, keepProb, config, reuse = False, phaseTrain = None, freeze = False):
    tuneArray = []
    imageSize = config.IMAGE_SIZE
    numInitChannel = config.NUM_RGB_CHANNEL
    conv2dList = config.conv2dList
    wscale = config.WSCALE
    h = tf.reshape(imagePh, [-1, imageSize[0], imageSize[1], numInitChannel])

    for layer in conv2dList:
#        print(klass.name)
#        print(h.shape)
        h = layer.apply(tuneArray, h, wscale, phaseTrain, keepProb, reuse, freeze)
    return (h, tuneArray)

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
#    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, baseConfig.floatSize))
    return accuracy

def accuracyML(logits, labels, k):
    logitsK = tf.nn.top_k(logits, k).indices
    labelsK = tf.nn.top_k(labels, k).indices
    accuracy = tf.size(tf.sets.set_intersection(logitsK,labelsK).values)
    return tf.cast(accuracy, baseConfig.floatSize) / tf.cast(k, baseConfig.floatSize)

def accuracyMLNth(logits, labels, n):
    correct_prediction = tf.equal(tf.argmax(logits[:,n:n+2], 1), tf.argmax(labels[:,n:n+2], 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, baseConfig.floatSize))
    return accuracy

def compile(images, labels, keepProb, isTrain, config, learning_rate = 1e-4):
    with tf.name_scope("tower_0"):
        logits, _ = inference(images, keepProb, config, False, isTrain)
    loss_value = loss(logits, labels)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_value)
    if config.dataType == "multiLabel":
        if isinstance(config.accuracy, tuple):
            method, arg = config.accuracy
            if method == "nth":
                acc_op = accuracyMLNth(logits, labels, arg)
            else:
                raise Exception("config.accuracy is not known")
        else:
            k = len(config.NUM_CLASSES_LIST)
            acc_op = accuracyML(logits, labels, k)
            #        acc_op = loss_value
    else:
        acc_op = accuracy(logits, labels)
    return (train_op, acc_op)

class Placeholders():
    def __init__(self, config, is_training = False):
        self.imagesPh = tf.placeholder(baseConfig.floatSize, shape=(None, config.IMAGE_SIZE[0]*config.IMAGE_SIZE[1]*config.NUM_RGB_CHANNEL))
        self.labelsPh = tf.placeholder(baseConfig.floatSize, shape=(None, config.NUM_CLASSES))
        self.keep_prob = tf.placeholder(baseConfig.floatSize)
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

def average_gradients(tower_grads):
    average_grads = []
    # 各variableに対して
    for grad_and_vars in zip(*tower_grads):
        grads = []
        
        # 各GPUの結果に対して 平均を取る
        for g, _ in grad_and_vars:
            # batch_normalization の updateは mean_var_with_updateで行っているので無視したい。
            # 同変数はreuse = None にしているので、 None としてリストされる。
            # None 以外でリストされてしまう変数も trainable = Falseなので大丈夫っぽい
            if g is not None:
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
def multiGpuLearning(config, phs, learning_rate= 1e-4):
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
                slicedImagePh = tf.slice(phs.getImages(), slice_start, [in_batch_length, config.IMAGE_SIZE[0] * config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL])
                slicedLabelPh = tf.slice(phs.getLabels(), slice_start, [in_batch_length, config.NUM_CLASSES])
                with tf.device("/gpu:"+str(gpu_id)):
                    with tf.name_scope("tower_"+str(gpu_id)):
                        logits, tuneArray = inference(slicedImagePh, phs.getKeepProb(), config, reuseVar, phs.getPhaseTrain())
                        # 1回目はinferenceでreuse(参照)しない。 createする
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

            if config.dataType == "multiLabel":
                if isinstance(config.accuracy, tuple):
                    method, arg = config.accuracy
                    if method == "nth":
                        acc_op = tf.add_n([accuracyMLNth(logits, labels, arg) for logits, labels in logitsList])
                    else:
                        raise Exception("config.accuracy is not known")
                else:
                    k = len(config.NUM_CLASSES_LIST)
                    acc_op = tf.add_n([accuracyML(logits, labels, k) for logits, labels in logitsList])
            else:
                acc_op = tf.add_n([accuracy(logits, labels) for logits, labels in logitsList])
            return train_op, acc_op, xtuneArray, debug
