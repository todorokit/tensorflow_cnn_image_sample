import tensorflow as tf
import config.baseConfig as baseConfig
from config.classes import classList
from util.utils import *

def img2op(path, config):
    contents = tf.read_file(path)
    img = tf.image.decode_image(contents)
    shape = tf.shape(img)
    height = tf.cond(shape[0] > shape[1], lambda : shape[1], lambda :shape[0])
    img = tf.image.resize_image_with_crop_or_pad(img, height, height)
    img = tf.image.resize_images(img, [config.IMAGE_SIZE[1],config.IMAGE_SIZE[0]])
    if baseConfig.dataFormat == "channels_first":
        img = tf.transpose(img, perm=[0, 3, 1, 2])
    imgop = tf.scalar_mul(1/255.0, tf.cast(tf.reshape(img, [config.IMAGE_SIZE[0]* config.IMAGE_SIZE[1]* config.NUM_RGB_CHANNEL]), baseConfig.floatSize))
    return imgop
    
def runTopnPb(sess, pbfile, imageop, config, thre):
    image = sess.run(imageop)
    with open(pbfile, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name="")
    arr = sess.run('fp32_storage/tower_0/fc/fc_var/fc_act:0', feed_dict={'input_image:0': [image], 'keep_prob:0': 1.0, 'phase_train:0': False})[0]
    ret = []
    if config.dataType == "multi-label":
        print("multi label")
        indices = filter(lambda x: arr[x] > thre, range(config.NUM_CLASSES))
    else:
        print("single lable top5")
        indices = top5(arr)
    for j in indices:
        ret.append((classList[j] , arr[j]))
    return ret
