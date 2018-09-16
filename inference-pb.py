import tensorflow as tf
from util.Container import getContainer
import dataset.SingleData
from util.utils import *
from config.classes import classList
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('config', "config.celeba", 'config module(file) name (no extension).')

#def inferenceAndSave(batch, images):
#    arrs = sess.run(logits, feed_dict={images_placeholder: images, phaseTrain: False, keepProb: 1.0})
#    i = 0
#    for arr in arrs:
#        k, real_image, path, dir = batch[i]
#        ans = top1(arr)
#        parentDir = os.path.join(cwd, FLAGS.outdir)
#        if ans and arr[ans] > FLAGS.other_score:
#            destDir = os.path.join(parentDir, classList[ans].replace(" ", "_"))
#        else:
#            destDir = os.path.join(parentDir, "other")
#
#        file    = os.path.basename(path)
#        if dir is None:
#            filename, extension = os.path.splitext(file)
#        else:
#            file_filename, extension = os.path.splitext(file)
#            filename = dir + "_" + file_filename
#        if k > 0:
#            dest = os.path.join(destDir , "%s-%05d%s"% (filename, k, extension))
#        else:
#            dest = os.path.join(destDir , "%s%s"% (filename,  extension))
#        os.makedirs(destDir, exist_ok=True)
#        cv2.imwrite(dest, real_image)
#        i += 1

def main(argv):
    if len(argv) < 3:
        print("usage: " + sys.argv[0] + " pbfile imgfile")
        sys.exit()
    pbfile = argv[1]
    path = argv[2]
    Container = getContainer(FLAGS)
    config = Container.get("config")

    with tf.Graph().as_default():
        imgOp = dataset.SingleData.img2op(path, config)
        sess = tf.Session()
        ret = dataset.SingleData.runTopnPb(sess, pbfile, imgOp, config, 0.7)
        for klass, score in ret:
            print("%s %g" %(klass, score))

if __name__ == '__main__':
    tf.app.run()
            
