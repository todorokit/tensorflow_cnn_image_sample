import os, sys
import tensorflow as tf

class MySaver():
    def __init__(self, sess, config, doRestore = True):
        self.saver = tf.train.Saver()
        self.sess = sess
        self.modelFile = config.modelFile
        if doRestore and os.path.exists(self.modelFile+".data-00000-of-00001") and os.path.exists(self.modelFile+".meta") and os.path.exists(self.modelFile+".index"):
            print("Restore " + self.modelFile)
            cwd = os.getcwd()
            self.saver.restore(self.sess, os.path.join(cwd, self.modelFile))

    def save(self, str1 = None):
        if str1 is not None:
            print("save: %s" % str1)
            sys.stdout.flush()
        cwd = os.getcwd()
        self.saver.save(self.sess, os.path.join(cwd, self.modelFile))
