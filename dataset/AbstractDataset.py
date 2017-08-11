class AbstractDataset():
    def getAccBatchSize(self):
        return self.acc_batch_size
    
    def accuracyFlow(self):
        return [(self.splitedAccuracyImage[i], self.splitedAccuracyLabel[i]) for i in range(self.numAccuracyLoop)]
    
    def getLen(self):
        return len(self.images)

    def calcAccuracy(self, sess, op, phs):
        acc_sum = 0
        for trains, labels in self.accuracyFlow():
            acc_sum += sess.run(op, feed_dict=phs.getDict(trains, labels, 1.0))
        return acc_sum / self.getLen()
