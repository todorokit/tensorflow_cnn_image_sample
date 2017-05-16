import os
import cv2
import numpy as np
import shutil

def loadImages(labelFilePath, imageSize, numClass):
    file = open(labelFilePath, 'r')
    image = []
    label = []
    paths = []
    for line in file:
        imgpath, labelIndex= line.rstrip().split()
        img = cv2.imread(imgpath)
        if img is None:
            continue
        img = makeImage(img, imageSize)
        image.append(img)
        labelData = np.zeros(numClass)
        if (labelIndex != "-1"):
            labelData[int(labelIndex)] = 1
        label.append(labelData)
        paths.append(imgpath)
    file.close()
    return (np.asarray(image), np.asarray(label), paths)

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
                                     minSize = (32, 32))
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
        for img, face in detectAnimeFace(imagePaths[i]):
            real_image.append(img)
            targets.append(makeImage(img, imageSize))
            faces.append(face)
    return (np.asarray(targets), real_image, faces)

def detectFace(filename, cascade_file = "haarcascade_frontalface_default.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (32, 32))
    ret = []
    for face in faces:
        x, y , w, h = face
        ret.append(image[y:y+h, x:x+w])
    return ret

def getFace(imagePaths, imageSize):
    targets = []
    for i in range(0, len(imagePaths)):
        for img in detectFace(imagePaths[i]):
            targets.append(makeImage(img, imageSize))
    return np.asarray(targets)

def backup(modelFile, backupDir, suffix = ""):
    cwd = os.getcwd()
    
    files = [modelFile+".data-00000-of-00001", modelFile + ".index", modelFile + ".meta", "checkpoint", "config.py"]

    backupDirPath=os.path.join(cwd, backupDir)
    os.makedirs(backupDirPath, exist_ok=True)
    for file1 in files:
        dest = os.path.join(backupDirPath, file1 + suffix)
        shutil.copyfile(os.path.join(cwd, file1), dest)

