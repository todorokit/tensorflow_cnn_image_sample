import tensorflow as tf
import tensorflow.python.platform

import config

NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
NUM_RGB_CHANNEL = config.NUM_RGB_CHANNEL

CHANNEL_MULTIPLIER_1 = config.CHANNEL_MULTIPLIER_1
FILTER1_SIZE = config.FILTER1_SIZE
IMAGE_SIZE_1 = IMAGE_SIZE  // 2

CHANNEL_MULTIPLIER_2 = config.CHANNEL_MULTIPLIER_2
FILTER2_SIZE = config.FILTER2_SIZE
IMAGE_SIZE_2 = IMAGE_SIZE_1 // 2

FC_MULTIPLIER_1 = config.FC_MULTIPLIER_1

def inference(images_placeholder, keep_prob):
    """ 予測モデル keep_probはトレーニング時以外は1.0にする    """
    
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを0.1で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_RGB_CHANNEL])
    # output shape [batch_size  , IMAGE_WIDTH, IMAGE_HEIGHT, NUM_RGB_CHANNEL]

    # 畳み込み層1
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([FILTER1_SIZE, FILTER1_SIZE, NUM_RGB_CHANNEL, CHANNEL_MULTIPLIER_1])
        b_conv1 = bias_variable([CHANNEL_MULTIPLIER_1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # output shape [batch_size  , IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL_MULTIPLIER_1]

    # プーリング層1
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    # output shape [batch_size  , IMAGE_WIDTH//2, IMAGE_HEIGHT//2, CHANNEL_MULTIPLIER_1]
    
    # 畳み込み層2
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([FILTER2_SIZE, FILTER2_SIZE, CHANNEL_MULTIPLIER_1, CHANNEL_MULTIPLIER_2])
        b_conv2 = bias_variable([CHANNEL_MULTIPLIER_2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # output shape [batch_size  , IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL_MULTIPLIER_2]

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    # output shape [batch_size  , IMAGE_WIDTH//4, IMAGE_HEIGHT//4, CHANNEL_MULTIPLIER_2]

    # 全結合層1
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([IMAGE_SIZE_2*IMAGE_SIZE_2*CHANNEL_MULTIPLIER_2, FC_MULTIPLIER_1])
        b_fc1 = bias_variable([FC_MULTIPLIER_1])
        h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_SIZE_2*IMAGE_SIZE_2*CHANNEL_MULTIPLIER_2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # output shape [batch_size  , FC_MULTIPLIER_1]

    # 全結合層2
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([FC_MULTIPLIER_1, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # output shape [batch_size, NUM_CLASSES]

    return y_conv

def loss(logits, labels):
    """ 誤差関数: 最適化の評価用に交差エントロピーの計算をする """
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.summary.scalar("cross_entropy", cross_entropy) # for tensorboard
    return cross_entropy

def training(loss, learning_rate):
    """ トレーニング: モデルの変数をlossを目安に最適化する """
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数 """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy) # for tensorboard
    return accuracy
