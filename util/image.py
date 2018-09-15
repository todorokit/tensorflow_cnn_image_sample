import os, re

import cv2
import numpy as np

import config.classes
from config import baseConfig

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
