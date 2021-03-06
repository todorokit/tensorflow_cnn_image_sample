# -*- coding: utf-8 -*-

import os

from util.utils import *
from util.MySaver import MySaver
import modelcnn
from dataset.LargeDataset import LargeDataset
from dataset.MultilabelLargeDataset import MultilabelLargeDataset

class DIContainer:
    def __init__(self, c):
        self.shared = {"Saver":True, "Sess": True, "Config": True, "Placeholders": True}
        self.objects = dict()
        self.componentFactory = c
        c.accept(self)

    def get(self, name):
        name = name.title()
        if name in self.shared:
            if not name in self.objects:
                self.objects[name] = self.componentFactory.get(name)
        else:
            return self.componentFactory.get(name)
        return self.objects[name]


class ComponentFactory:
    def __init__(self):
        self.container = None

    def get(self, name):
        method_name = 'build' + name
        return getattr(self, method_name)()

    def accept(self, c):
        self.container = c

class MyFactory(ComponentFactory):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.config = importer(FLAGS.config)
        pass

    def buildConfig(self):
        return self.config

    def buildFlags(self):
        return self.FLAGS
    
    def buildSess(self):
        config = self.container.get("config")
        return makeSess(self.FLAGS, config)

    def buildSaver(self):
        config = self.container.get("config")
        sess = self.container.get("sess")
        return MySaver(sess, config)

    def buildSaver_No_Restore(self):
        config = self.container.get("config")
        sess = self.container.get("sess")
        return MySaver(sess, config, False)
    
    def buildPlaceholders(self):
        return modelcnn.Placeholders(self.config, True)

    def buildOps(self):
        phs = self.container.get("placeholders")
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return modelcnn.compile(phs, config, FLAGS)

    def buildOps_Mgpu(self):
        phs = self.container.get("placeholders")
        config = self.container.get("config")
        return modelcnn.multiGpuLearning(config, phs)

    def buildLogits(self):
        phs = self.container.get("placeholders")
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return (phs, modelcnn.inference(phs, config,  FLAGS))

    def buildLargetraindataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return LargeDataset(config.trainFile, config, FLAGS.batch_size, config.isCacheTrain)

    def buildLargetestdataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return LargeDataset(config.testFile, config, FLAGS.acc_batch_size, config.isCacheTest)

    def buildValiddataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        if config.validFile and os.path.exists(config.validFile):
            validDataset = LargeDataset(config.validFile, config, FLAGS.acc_batch_size, config.isCacheTest)
        else:
            validDataset = None
        return validDataset
    
    def buildMultilargetraindataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return MultilabelLargeDataset(config.trainFile, config, FLAGS.batch_size, config.isCacheTrain)

    def buildMultilargetestdataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return MultilabelLargeDataset(config.testFile, config, FLAGS.acc_batch_size, config.isCacheTest)

    def buildMultivaliddataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        if config.validFile and os.path.exists(config.validFile):
            validDataset = MultilabelLargeDataset(config.validFile, config, FLAGS.acc_batch_size, config.isCacheTest)
        else:
            validDataset = None
        return validDataset

def getContainer(FLAGS):
    return DIContainer(MyFactory(FLAGS))
