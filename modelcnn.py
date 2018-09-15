import tensorflow as tf

from config import baseConfig
from tensorflow.contrib.all_reduce.python import all_reduce

def model(imagePh, keepProb, config, reuse = False, phaseTrain = None, freeze = False):
    tuneArray = []
    h = tf.reshape(imagePh, [-1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_RGB_CHANNEL])

    for layer in config.conv2dList:
#        print(klass.name)
#        print(h.shape)
        h = layer.apply(tuneArray, h, phaseTrain, keepProb, reuse, freeze)
    return (h, tuneArray)

def loss(logits, labels):
    loss = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0-1e-10)))
#    with tf.device("/cpu:0"):
#        tf.summary.scalar("loss", loss)
    return loss

def crossentropy(logits, labels):
    # 精度が低い
    # tf.Session().run(tf.constant(1.0) - (tf.constant(1.0) - 1e-10)) => 0
    logits = tf.clip_by_value(logits,1e-6,1.0 - 1e-6)
    loss = tf.reduce_sum(labels* -tf.log(logits) + (1 - labels) *  -tf.log(1 - logits))
#    with tf.device("/cpu:0"):
#        tf.summary.scalar("loss", loss)
    return loss

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, baseConfig.floatSize))
    return accuracy

def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale
            for grad in tf.gradients(loss * loss_scale, variables)]

DEFAULT_DTYPE = tf.float32
def float32_variable_storage_getter(getter, name, shape=None, dtype=DEFAULT_DTYPE,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    #print(name, dtype, trainable)
    if dtype != tf.float32:
        variable = getter(name, shape, dtype=tf.float32,
                          initializer=initializer, regularizer=regularizer,
                          trainable=trainable,
                          *args, **kwargs)
        variable = tf.cast(variable, dtype)
    else:
        variable = getter(name, shape, dtype=dtype,
                          initializer=initializer, regularizer=regularizer,
                          trainable=trainable,
                          *args, **kwargs)
    return variable

def inference(images, keepProb, isTrain, config, FLAGS):
    with tf.device('/cpu:0'), \
         tf.variable_scope('fp32_storage', custom_getter=float32_variable_storage_getter):
        with tf.name_scope("tower_0"):
            logits, _ = model(images, keepProb, config,  False, isTrain)
            return logits

def compile(images, labels, keepProb, isTrain, config, FLAGS):
    learning_rate = FLAGS.learning_rate
    momentum      = 0.9
    loss_scale    = 128
    
    with tf.device('/gpu:0'), \
         tf.variable_scope('fp32_storage', custom_getter=float32_variable_storage_getter):
        with tf.name_scope("tower_0"):
            logits, _ = model(images, keepProb, config, False, isTrain)
            if config.dataType == "multi-label":
                loss_value = crossentropy(logits, labels)
            else:
                loss_value = loss(logits, labels)
# MomentumOptimizer fp32でも収束しなかった。
#        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#        grads = gradients_with_loss_scaling(loss_value, variables, loss_scale)
#        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
#        train_op = optimizer.apply_gradients(zip(grads, variables))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_value)

    if config.dataType == "multi-label":
        acc_op = loss_value
    else:
        acc_op = accuracy(logits, labels)
    return (train_op, acc_op)

class Placeholders():
    def __init__(self, config, is_training = False):
        self.imagesPh = tf.placeholder(baseConfig.floatSize, shape=(None, config.IMAGE_SIZE[0]*config.IMAGE_SIZE[1]*config.NUM_RGB_CHANNEL))
        ## 結果はfloat32
        self.labelsPh = tf.placeholder(tf.float32, shape=(None, config.NUM_CLASSES))
        self.keep_prob = tf.placeholder(tf.float32)
#        self.keep_prob = tf.placeholder(baseConfig.floatSize)
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
        lossvalues = []
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
                with tf.device("/gpu:"+str(gpu_id)), \
                     tf.variable_scope('fp32_storage', custom_getter=float32_variable_storage_getter):
                    with tf.name_scope("tower_"+str(gpu_id)):
                        logits, tuneArray = model(slicedImagePh, phs.getKeepProb(), config, reuseVar, phs.getPhaseTrain())
                        # 1回目はmodelでreuse(参照)しない。 allocateする
                        # 2回目以降はreuse(参照)する。
                        reuseVar = True
                        if xtuneArray is None:
                            xtuneArray = tuneArray
                        logitsList.append((logits, slicedLabelPh))
                        if config.dataType == "multi-label":
                            loss_value = crossentropy(logits, slicedLabelPh)
                        else:
                            loss_value = loss(logits, slicedLabelPh)
                        grads = opt.compute_gradients(loss_value)
                        towerGrads.append(grads)
                        lossvalues.append(loss_value)

            grads = average_gradients(towerGrads)
            train_op = opt.apply_gradients(grads, global_step=global_step)

            if config.dataType == "multi-label":
                acc_op = tf.add_n(lossvalues)
            else:
                acc_op = tf.add_n([accuracy(logits, labels) for logits, labels in logitsList])
            return train_op, acc_op, xtuneArray, debug
