import os, sys, re, random

#NUM_IMAGE=202599
NUM_IMAGE=195148
trainFile="train.txt"
testFile="test.txt"
attrFile = "/data/celeba/list_attr_celeba.txt"
imageDir = "/data/celeba/img_align_celeba/"
if len(sys.argv) >= 2 and sys.argv[1].isnumeric():
    trainNum = int(sys.argv[1])
else:
    trainNum = 160000
if len(sys.argv) >= 3 and sys.argv[2].isnumeric():
    testNum = int(sys.argv[2])
else:
    testNum  = 30000

def find(dir, dirs):
    ret = dirs
    for file in os.listdir(dir):
        if file == "." or file == "..":
            continue
        realfile = file
        ret.append(realfile)
    return ret

def main(trainNum, testNum):
    attrFp = open(attrFile, "r")
    attr = {}
    classes = {}
    for line in attrFp:
        line = line.strip().replace("  ", " ")
        if re.search("^\d+.jpg ",line):
            words = line.split(" ")
            file = words.pop(0)
            attr[file] = ",".join(words).replace("-1", "0")
        else:
            words = line.split(" ")
            ix = 0
            for word in words:
                classes[ix*2] = "not " + word
                classes[ix*2+1] = word
                ix += 1
            fp = open(os.path.join("config", "classes.py"), "w")
            fp.write("classList = {}\n")
            for ix in classes:
                fp.write("classList[%d] = \"%s\"\n" % (ix, classes[ix]))


    lis = find(imageDir, [])
    random.shuffle(lis)
    fp = open(trainFile, "w")
    for i in range(trainNum):
        file = lis.pop()
        fp.write(os.path.join(imageDir, "%s,%s\n" % (file, attr[file]) ))
    fp.close()

    fp = open(testFile, "w")
    for i in range(testNum):
        file = lis.pop()
        fp.write(os.path.join(imageDir, "%s,%s\n" % (file, attr[file]) ))
    fp.close()

main(trainNum, testNum)
