import os, sys, time, re

import cv2

import deeptool

sameImages= []
cachedImages = None
def isSameImage(imghist, checkhist):
    ret = cv2.compareHist(imghist, checkhist, 0)
    ret2 = cv2.compareHist(imghist, checkhist, 1)
    ret3 = cv2.compareHist(imghist, checkhist, 2)
    ret4 = cv2.compareHist(imghist, checkhist, 3)
    return ret > 0.995 and ret2 < 300, (ret, ret2, ret3, ret4)

def eraseByHist(file):
    global sameImages, cachedImages
    img = cv2.imread(file)
    img = cv2.resize(img, (64, 64))
    imghist = cv2.calcHist([img], [0, 1, 2], None,  [8, 8, 8], [0,256,0,256,0,256])
    reg = re.compile(r"(\d+).(png|jpe?g)$")
    sys.stdout.flush()
    m = reg.search(file)
    fileId = int(m.group(1))
    if cachedImages is None:
        cachedImages = {file: (imghist, fileId)}
    else:
        appended = False
        score = (0, 0 , 0, 0)
        for cachedfile in cachedImages:
#            sys.stdout.write ("*")
#            sys.stdout.flush()
            cachedImgHist, cachedFileId = cachedImages[cachedfile]
            isSame , score= isSameImage(imghist, cachedImgHist)
            if isSame and fileId < cachedFileId + 210:
#                sys.stdout.write ("!")
#                sys.stdout.flush()
                sameImages.append((file, cachedfile, score))
                appended = True
                break
        if not appended:
            cachedImages[file] = (imghist, fileId)
        

def main(args):
    global sameImages, cachedImages
    if len(args) == 1:
        print(args[0]+" dirname")
        exit()
    cachedImages = None
    if os.path.isdir(args[1]):
        start = time.time()
        i = 0
        for file in deeptool.listDir(args[1]):
            if os.path.isdir(file):
                print("--"+file+"--")
                sys.stdout.flush()
                cachedImages = None
                for file2 in deeptool.listDir(file):
                    eraseByHist(file2)
                    i = i + 1
                    if (i % 1000 == 0):
                        end = time.time()
                        print (" %g data / sec" % ( 1000 / (end - start) ))
                        start = end
            else:
                eraseByHist(file)
                i = i + 1
                if (i % 1000 == 0):
                    end = time.time()
                    print (" %g data / sec" % ( 1000 / (end - start) ))
                    start = end
    else:
        print(args[0] + " dirname")
        exit()
    print("--------------------------------")
    for file, matchfile,score in sameImages:
        os.unlink(file)
        print (file, matchfile, score)


main(sys.argv)
