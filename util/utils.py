import os, shutil

import tensorflow as tf

from config import baseConfig

flags = tf.app.flags
flags.DEFINE_float('memory', 0.95, 'Using gpu memory.')

def backup(configFileName, modelFile, backupDir, suffix = ""):
    cwd = os.getcwd()
    
    files = [modelFile+".data-00000-of-00001", modelFile + ".index", modelFile + ".meta", "checkpoint", configFileName]

    backupDirPath=os.path.join(cwd, backupDir)
    os.makedirs(backupDirPath, exist_ok=True)
    for file1 in files:
        dest = os.path.join(backupDirPath, os.path.basename(file1) + suffix)
        shutil.copyfile(os.path.join(cwd, file1), dest)

def listDir(dir):
    ret = []
    for file in os.listdir(dir):
        if file == "." or file == "..":
            continue;
        ret.append(os.path.join(dir, file))
    return ret


def makeSess(flags):
    if flags.memory == 0.0:
        sess = tf.Session()
    else:
        gpuConfig = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=flags.memory),
        )
        sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    return sess

#    memoryが少ないと誤判定 
#    def buildSess_Inference(self):
#        return makeSessInference()
#def makeSessInference():
#    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#    sess = tf.Session(config=config)
#    sess.run(tf.global_variables_initializer())
#    return sess

def saveBest(config, FLAGS, sess, mysaver, score):
    def getBest():
        path = os.path.join("best-model", config.scoreFileName)
        if os.path.exists(path):
            fp = open(path, "r")
            score = fp.read().replace("\n", "")
            fp.close()
            return float(score)
        else:
            return 0.0

    def writeScore(path, score):
        fp = open(path, "w")
        fp.write(str(score))
        fp.close()
        
    if getBest() < float(score):
        print("Save Best ")
        cwd = os.getcwd()
        mysaver.save()
        backup(deepImportToPath(FLAGS.config), config.modelFile, "best-model")
        writeScore(os.path.join("best-model", config.scoreFileName), score)

def printCUDA_env():
    print ("------------GPU DEVICES----------------")
    try:
        print (os.environ["CUDA_VISIBLE_DEVICES"])
    except:
        print ("all gpu (default)")
    print ("---------------------------------------")

def importer(name, root_package=False, relative_globals=None, level=0):
    """ We only import modules, functions can be looked up on the module.
    Usage: 

    from foo.bar import baz
    >>> baz = importer('foo.bar.baz')

    import foo.bar.baz
    >>> foo = importer('foo.bar.baz', root_package=True)
    >>> foo.bar.baz

    from .. import baz (level = number of dots)
    >>> baz = importer('baz', relative_globals=globals(), level=2)
    """
    return __import__(name, locals=None, # locals has no use
                      globals=relative_globals, 
                      fromlist=[] if root_package else [None],
                      level=level)

def deepImportToPath (str) :
    if os.name == 'nt':
        return str.replace(".", "\\")+".py"
    else:
        return str.replace(".", "/")+".py"

def top5(arr):
    return arr.argsort()[-5:][::-1]

def top1(arr):
    return arr.argsort()[-1:][::-1][0]
