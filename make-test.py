import os
import sys
import random

sourceDir = 'img'

trFile = 'train.txt'
teFile = 'test.txt'
mapFile = 'config2.py'

if len(sys.argv) != 3:
    print ("usage %s trainNum testNum" % (sys.argv[0]))
    exit()
datanum = int(sys.argv[1])
testnum = int(sys.argv[2])

def listClass(dir):
    ret = []
    for file in os.listdir(dir):
        if(file == "." or file == ".." or file == "-1"):
            continue;
        ret.append(file)
    return ret

def find(dir, dirs):
    ret = dirs
    for file in os.listdir(dir):
        realfile = os.path.join("%s","%s")%(dir,file)
        if (os.path.isdir(realfile)):
            ret = find(realfile, ret)
        else:
            ret.append(realfile)
    return ret

def ref(dict, key, default):
    try:
        return dict[key]
    except:
        return default

def addDict(dict, key):
    try:
        dict[key] += 1
    except:
        dict[key] = 1

dirs = listClass(sourceDir)
def getId(className):
    return dirs.index(className)

images = find(sourceDir, [])
random.shuffle(images);

fp = open(mapFile, "w")
fp.write("classList = {}\n")
i = 0
for className in dirs:
    fp.write("classList[%d] = \"%s\"\n"% (i, className))
    i += 1
fp.close()

teFp = open(teFile, "w")
trFp = open(trFile, "w")

limits = {};
limits2 = {};
for image in images:
    className = os.path.basename(os.path.dirname(image))
    isTest = False
    if ref(limits2, className, 0) >= testnum:
        continue
    elif ref(limits, className, 0) >= datanum:
        addDict(limits2, className)
        isTest = True
    else:
        addDict(limits, className)

    if className == "-1":
        # 損失関数も正当率計算もzeroベクトルに対応していない。
        id = -1;
        continue

    else:
        id = getId(className);

    if isTest:
        teFp.write("%s %d\n" % (image, id));
    else:
        trFp.write("%s %d\n" % (image, id));

trFp.close()
teFp.close()
