import os, re

import cv2
import numpy as np

import config.classes
from config import baseConfig

def loadMultiLabelImages(labelFilePath, imageSize, numClassList, imsgeResize):
    file = open(labelFilePath, 'r')
    image = []
    label = []
    paths = []
    numClass = sum(numClassList)
    loop = 0
    for line in file:
        line = line.rstrip()
        words = line.split()
        imgpath = words.pop(0)
        lineLabels = [int(word) for word in words]
        
        img = cv2.imread(imgpath)
        if img is None:
            continue
        img = makeImage(img, imageSize, imsgeResize)
        image.append(img)
        labelData = np.astype(lineLabels, dtype=np.int8)
        label.append(labelData)
        paths.append(imgpath)

        loop += 1
        if loop % 1000 == 0:
            print("load %d" % (loop,))

    file.close()
    return (np.asarray(image), np.asarray(label), paths)

def loadImages(labelFilePath, imageSize, numClass, imageResize):
    file = open(labelFilePath, 'r')
    image = []
    label = []
    paths = []
    for line in file:
        line = line.rstrip()
        match = re.search(r"^valid[/\\](\w+)[/\\][\w.-]+$", line)
        if match:
            imgpath = line
            className = match.group(1)
            labelIndex = [k for (k, v) in config.classes.classList.items() if v == className][0]
        else:
            imgpath, labelIndex = line.split(",")
        
        img = cv2.imread(imgpath)
        if img is None:
            continue
        img = makeImage(img, imageSize, imageResize)
        image.append(img)
        labelData = np.zeros(numClass)
        labelData[int(labelIndex)] = 1
        label.append(labelData)
        paths.append(imgpath)
    file.close()
    return (np.asarray(image), np.asarray(label), paths)

def makeImage(img, imageSize, resize = "resize"):
    if resize == "resize":
        img = cv2.resize(img, (imageSize[1], imageSize[0]))
    elif resize == "crop":
        # fixme  pad version
        h, w, c = img.shape
        y = (h - imageSize[1]) // 2
        x = (w - imageSize[0]) // 2
        img = img[y:y+imageSize[1], x:x+imageSize[0]]
    else:
        raise Exception("invalid resize type")
    return img.flatten().astype(baseConfig.npFloatSize)/255.0

# https://github.com/nagadomi/lbpcascade_animeface
def detectAnimeFace(filename, imageSize = (73,73), cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = imageSize)
    ret = []
    ret2 = []
    for face in faces:
        x, y , w, h = face
        ret.append(image[y:y+h, x:x+w])
        ret2.append(face)
    return zip(ret, ret2)

def getAnimeFace(imagePaths, imageSize):
    targets = []
    real_image = []
    faces = []
    for i in range(0, len(imagePaths)):
        for img, face in detectAnimeFace(imagePaths[i], imageSize):
            real_image.append(cv2.imread(imagePaths[i]))
            targets.append(makeImage(img, imageSize))
            faces.append(face)
    return (np.asarray(targets), real_image, faces)

def detectFace(filename, imageSize = (73,73), cascade_file = "haarcascade_frontalface_default.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 1,
                                     minSize = imageSize)
    ret = []
    ret2 = []
    for face in faces:
        x, y , w, h = face
        ret.append(image[y:y+h, x:x+w])
        ret2.append(face)
    return zip(ret, ret2)

def getFace(imagePaths, imageSize):
    targets = []
    real_image = []
    faces = []
    for i in range(0, len(imagePaths)):
        for img, face in detectFace(imagePaths[i], imageSize):
            real_image.append(img)
            targets.append(makeImage(img, imageSize))
            faces.append(face)
    return (np.asarray(targets), real_image, faces)

def getImage(imagePaths, imageSize):
    targets = []
    real_image = []
    faces = []
    for i in range(0, len(imagePaths)):
        img = cv2.imread(imagePaths[i])
        real_image.append(img)
        targets.append(makeImage(img, imageSize))
        faces.append([])
    return (np.asarray(targets), real_image, faces)
