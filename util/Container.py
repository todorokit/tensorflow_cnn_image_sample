# -*- coding: utf-8 -*-

import os

from util.utils import *
from util.MySaver import MySaver
import modelcnn
from dataset.InMemoryDataset import InMemoryDataset 
from dataset.InMemoryDatasetForTest import InMemoryDatasetForTest

flags = tf.app.flags
FLAGS = flags.FLAGS

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
        return makeSess(self.FLAGS)

    def buildSaver(self):
        config = self.container.get("config")
        sess = self.container.get("sess")
        return MySaver(sess, config)

    def buildPlaceholders(self):
        return modelcnn.Placeholders(self.config, True)

    def buildOps(self):
        phs = self.container.get("placeholders")
        config = self.container.get("config")
        return modelcnn.compile(phs.getImages(), phs.getLabels(), phs.getKeepProb(), phs.getPhaseTrain(), config)

    def buildTraindataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return InMemoryDataset(config.trainFile, config, FLAGS.batch_size, FLAGS.acc_batch_size)

    def buildTestdataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        return InMemoryDatasetForTest(config.testFile, config, FLAGS.acc_batch_size)

    def buildValiddataset(self):
        config = self.container.get("config")
        FLAGS = self.container.get("flags")
        if config.validFile and os.path.exists(config.validFile):
            validDataset = InMemoryDatasetForTest(config.validFile, config, FLAGS.acc_batch_size)
        else:
            validDataset = None
        return validDataset

Container = DIContainer(MyFactory(FLAGS))
