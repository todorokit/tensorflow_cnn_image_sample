import tensorflow as tf
from util.Container import getContainer
import dataset.LargeDataset
from util.utils import *
from config.classes import classList
import sys

flags = tf.app.flags
flags.DEFINE_string('config', "config.celeba", 'config module(file) name (no extension).')
FLAGS = flags.FLAGS

def main(argv):
    if len(argv) == 1:
        print("usage: " + sys.argv[0] + " file")
        sys.exit()
    path = argv[1]
    Container = getContainer(FLAGS)
    config = Container.get("config")

    with tf.Graph().as_default():
        imgOp = dataset.LargeDataset.img2vector(path, config)
        sess = tf.Session()
        image = sess.run(imgOp)
        images = [image]
        with open('graph.pb', 'rb') as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())

            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
            tf.import_graph_def(graph_def, name="")
        arr = sess.run('fp32_storage/tower_0/fc/fc_var/fc_act:0', feed_dict={'input_image:0': images, 'keep_prob:0': 1.0, 'phase_train:0': False})[0]
        indices = top5(arr)
        for j in indices:
            print("%s %g" %(classList[j] , arr[j]))

if __name__ == '__main__':
    tf.app.run()
            
