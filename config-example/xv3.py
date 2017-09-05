import tensorflow as tf
import tensorflow.python.platform

import math
import wrappers as wr

NUM_CLASSES = 20
IMAGE_SIZE = (64, 64)
NUM_RGB_CHANNEL = 3

conv2dList = [ wr.Conv2D("conv_bn1", 32, (18, 18), padding="SAME"),
               wr.Conv2D("conv_bn1-a", 32, (1, 1), padding="SAME"),
               wr.Conv2D("conv_bn1-b", 32, (1, 1), padding="SAME"),
               wr.MaxPooling2D("pool1", padding="SAME"),
               wr.Conv2D("conv_bn2", 32, (7, 7), padding="SAME"),
               wr.Conv2D("conv_bn2-a", 32, (1, 1), padding="SAME"),
               wr.Conv2D("conv_bn2-b", 32, (1, 1), padding="SAME"),
               wr.MaxPooling2D("pool2", padding="SAME"),
               wr.Conv2D("conv_bn3", 64, (3, 3), padding="SAME"),
               wr.Conv2D("conv_bn3-a", 64, (1, 1), padding="SAME"),
               wr.Conv2D("conv_bn3-b", 64, (1, 1), padding="SAME"),
               wr.MaxPooling2D("pool3", padding="SAME"),
               wr.Flatten(),
               wr.Dropout(),
               wr.FullConnect("fc", NUM_CLASSES)
]

WSCALE=math.sqrt(2.0/NUM_CLASSES)
# multi gpu make same model.ckpt
modelFile = "model.ckpt"
num_gpu = 2

imageResize = "resize"
dataType = "label"
accuracy = None
faceType = "anime"

trainFile = "train.txt"
testFile = "test.txt"
validFile = "valid.txt"
