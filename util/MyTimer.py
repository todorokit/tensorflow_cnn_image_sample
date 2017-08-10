import time, datetime

class MyTimer ():
    def __init__(self):
        self.startTime = time.time()

    def getNow(self, format):
        return datetime.datetime.today().strftime(format)

    def getTime(self):
        return time.time() - self.startTime
