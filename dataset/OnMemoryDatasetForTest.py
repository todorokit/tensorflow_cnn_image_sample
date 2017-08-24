from dataset.AbstractDataset import AbstractDataset
import util.image

class OnMemoryDatasetForTest(AbstractDataset):
    def __init__(self, path, config, acc_batch_size):
        if config.dataType == "multiLabel":
            self.images, self.labels, self.paths= util.image.loadMultiLabelImages(path, config.IMAGE_SIZE, config.NUM_CLASSES_LIST, config.imageResize)
        else:
            self.images, self.labels, self.paths= util.image.loadImages(path, config.IMAGE_SIZE, config.NUM_CLASSES)
        self.acc_batch_size = acc_batch_size
        self.splitedAccracyImage = None
        self.splitedAccracyLabel = None
        self.splitedAccuracyPath = None
        self.numAccuracyLoop = 0
        self.calcBatchImage()
        
    def calcBatchImage(self):
        n = len(self.images)
        self.numAccuracyLoop = n//self.acc_batch_size
        self.splitedAccuracyImage = []
        self.splitedAccuracyLabel = []
        self.splitedAccuracyPath  = []
        batch = None
        for i in range(self.numAccuracyLoop):
            batch = self.acc_batch_size * i
            #print(batch, batch+self.acc_batch_size)
            self.splitedAccuracyImage.append(self.images[batch:batch+self.acc_batch_size])
            self.splitedAccuracyLabel.append(self.labels[batch:batch+self.acc_batch_size])
            self.splitedAccuracyPath.append(self.paths[batch:batch+self.acc_batch_size])

        if batch is None:
            batch = 0
        else:
            batch = batch + self.acc_batch_size
        if batch != n:
            self.numAccuracyLoop += 1
            #print(batch, n)
            self.splitedAccuracyImage.append(self.images[batch:n])
            self.splitedAccuracyLabel.append(self.labels[batch:n])
            self.splitedAccuracyPath.append(self.paths[batch:n])

    def flow(self):
        return [(self.splitedAccuracyImage[i], self.splitedAccuracyLabel[i], self.splitedAccuracyPath[i]) for i in range(self.numAccuracyLoop)]
            
    
            
