from dataset.AbstractDataset import AbstractDataset
import util.image

class InMemoryDatasetForTest(AbstractDataset):
    def __init__(self, path, config, acc_batch_size):
        self.images, self.labels, _= util.image.loadImages(path, config.IMAGE_SIZE, config.NUM_CLASSES)
        self.acc_batch_size = acc_batch_size
        self.splitedAccracyImage = None
        self.splitedAccracyLabel = None
        self.numAccuracyLoop = 0
        self.calcBatchImage()
        
    def calcBatchImage(self):
        n = len(self.images)
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

