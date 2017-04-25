import os
import cv2
import numpy as np

def loadImages(labelFilePath, imageSize, numClass):
    file = open(labelFilePath, 'r')
    image = []
    label = []
    for line in file:
        imgpath, labelIndex= line.rstrip().split()
        img = makeImage(cv2.imread(imgpath), imageSize)
        image.append(img)
        labelData = np.zeros(numClass)
        labelData[int(labelIndex)] = 1
        label.append(labelData)
    file.close()
    return (np.asarray(image), np.asarray(label))

def makeImage(img, imageSize):
    img = cv2.resize(img, (imageSize, imageSize))
    return img.flatten().astype(np.float32)/255.0

# https://github.com/nagadomi/lbpcascade_animeface
def detectAnimeFace(filename, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    ret = []
    for face in faces:
        x, y , w, h = face
        ret.append(image[y:y+h, x:x+w])
    return ret

def getAnimeFace(imagePaths, imageSize):
    targets = []
    for i in range(0, len(imagePaths)):
        for img in detectAnimeFace(imagePaths[i]):
            targets.append(makeImage(img, imageSize))
    return np.asarray(targets)
