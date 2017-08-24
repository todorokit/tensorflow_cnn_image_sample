from dataset.AbstractDataset import AbstractDataset
import util.image

class OnMemoryDataset(AbstractDataset):
    def __init__(self, path, config, batch_size, acc_batch_size):
        if config.dataType == "multiLabel":
            self.images, self.labels, self.paths= util.image.loadMultiLabelImages(path, config.IMAGE_SIZE, config.NUM_CLASSES_LIST, config.imageResize)
        else:
            self.images, self.labels, self.paths= util.image.loadImages(path, config.IMAGE_SIZE, config.NUM_CLASSES)
        self.batch_size = batch_size
        self.acc_batch_size = acc_batch_size
        self.splitedImage = None
        self.splitedLabel = None
        self.splitedAccracyImage = None
        self.splitedAccracyLabel = None
        self.numTrainLoop = 0
        self.numAccuracyLoop = 0
        self.calcBatchImage()
        
    def calcBatchImage(self):
        n = len(self.images)
        self.numTrainLoop = n // self.batch_size
        self.splitedImage = []
        self.splitedLabel = []
        batch = None
        # train は割り切って下さい
        for i in range(self.numTrainLoop):
            batch = self.batch_size * i
            self.splitedImage.append(self.images[batch:batch+self.batch_size])
            self.splitedLabel.append(self.labels[batch:batch+self.batch_size])
        if batch is None:
            batch = 0
        else:
            batch = batch + self.batch_size
        if batch != n:
            self.numTrainLoop += 1
            #print(batch, n)
            self.splitedImage.append(self.images[batch:n])
            self.splitedLabel.append(self.labels[batch:n])

        self.numAccuracyLoop = n//self.acc_batch_size
        self.splitedAccuracyImage = []
        self.splitedAccuracyLabel = []
        batch = None
        for i in range(self.numAccuracyLoop):
            batch = self.acc_batch_size * i
            #print(batch, batch+self.acc_batch_size)
            self.splitedAccuracyImage.append(self.images[batch:batch+self.acc_batch_size])
            self.splitedAccuracyLabel.append(self.labels[batch:batch+self.acc_batch_size])

        if batch is None:
            batch = 0
        else:
            batch = batch + self.acc_batch_size
        if batch != n:
            self.numAccuracyLoop += 1
            #print(batch, n)
            self.splitedAccuracyImage.append(self.images[batch:n])
            self.splitedAccuracyLabel.append(self.labels[batch:n])

    def flow(self):
        return [(self.splitedImage[i], self.splitedLabel[i]) for i in range(self.numTrainLoop)]

    def train(self, sess, op, phs, dropout=0.5):
        for trains, labels in self.flow():
            sess.run(op, feed_dict=phs.getDict(trains, labels, dropout, True))
