import os, sys, re, random

NUM_IMAGE=202599

trainFile="train.txt"
testFile="test.txt"
attrFile = "d:\\data\\celeba\\list_attr_celeba.txt"
imageDir = "d:\\data\\celeba\\img_align_celeba"
trainNum = 30000
testNum  = 5000

def main(trainNum, testNum):
    attrFp = open(attrFile, "r")
    attr = {}
    classes = {}
    for line in attrFp:
        line = line.strip().replace("  ", " ")
        if re.search("^\d+.jpg ",line):
            words = line.split(" ")
            file = words.pop(0)
            attr[file] = " ".join(words).replace("-1", "0")
        else:
            words = line.split(" ")
            ix = 0
            for word in words:
                classes[ix*2] = "not " + word
                classes[ix*2+1] = word
                ix += 1
            fp = open("config\\classes.py", "w")
            fp.write("classList = {}\n")
            for ix in classes:
                fp.write("classList[%d] = \"%s\"\n" % (ix, classes[ix]))
            
    lis = list(range(NUM_IMAGE))
    random.shuffle(lis)
    fp = open(trainFile, "w")
    for i in range(trainNum):
        id = lis.pop()+1
        file = "%06d.jpg" % (id,)
        fp.write(os.path.join(imageDir, "%s %s\n" % (file, attr[file]) ))
    fp.close()

    fp = open(testFile, "w")
    for i in range(testNum):
        id = lis.pop()+1
        file = "%06d.jpg" % (id,)
        fp.write(os.path.join(imageDir, "%s %s\n" % (file, attr[file]) ))
    fp.close()

main(trainNum, testNum)
