import math

import tensorflow as tf

from wrappers import Conv2D_bn, MaxPooling2D, AveragePooling2D, Flatten, Dropout, FullConnect, Concat, GlobalAveragePooling2D
from modelcnn import crossentropy

NUM_CLASSES = 40
IMAGE_SIZE = (73, 73)
NUM_RGB_CHANNEL = 3

conv2dList = [
    Conv2D_bn("conv1", 80, (1, 1)),
    Conv2D_bn("conv1-a", 192, (3,3)),
    MaxPooling2D("pool1"),
    Concat(
        [Conv2D_bn("conv2-a", 64, (1, 1), padding="SAME") ],
        [Conv2D_bn("conv2-b-1", 48, (1, 1), padding="SAME"), Conv2D_bn("conv2-b-2", 64, (5, 5), padding="SAME")],
        [Conv2D_bn("conv2-c-1", 64, (1, 1), padding="SAME"), Conv2D_bn("conv2-c-2", 96, padding="SAME"), Conv2D_bn("conv2-c-3", 96, padding="SAME")],
        [AveragePooling2D("apool2", (3,3), strides=(1,1), padding="SAME"), Conv2D_bn("conv2-d", 32, (1,1), padding="SAME") ],
        name = "concat2"
    ),
    Concat([Conv2D_bn("conv3-a", 64, (1, 1), padding="SAME") ],
           [Conv2D_bn("conv3-b-1", 48, (1, 1), padding="SAME"), Conv2D_bn("conv3-b-2", 64, (5, 5), padding="SAME")],
           [Conv2D_bn("conv3-c-1", 64, (1, 1), padding="SAME"), Conv2D_bn("conv3-c-2", 96, padding="SAME"), Conv2D_bn("conv3-c-3", 96, padding="SAME")],
           [AveragePooling2D("apool3", (3,3), strides=(1,1), padding="SAME"), Conv2D_bn("conv3-d", 64, (1,1), padding="SAME") ],
           name="concat3"
    ),
    Concat([Conv2D_bn("conv4-a", 64, (1, 1), padding="SAME") ],
        [Conv2D_bn("conv4-b-1", 48, (1, 1), padding="SAME"), Conv2D_bn("conv4-b-2", 64, (5, 5), padding="SAME")],
        [Conv2D_bn("conv4-c-1", 64, (1, 1), padding="SAME"), Conv2D_bn("conv4-c-2", 96, padding="SAME"), Conv2D_bn("conv4-c-3", 96, padding="SAME")],
        [AveragePooling2D("apool4", (3,3), strides=(1,1), padding="SAME"), Conv2D_bn("conv4-d", 64, (1,1), padding="SAME")]
         ,name = "concat4"
    ),
    Concat([Conv2D_bn("conv5-a", 384, (3, 3), strides=(2,2)) ],
        [Conv2D_bn("conv5-b-1", 64, (1, 1), padding="SAME"), Conv2D_bn("conv5-b-2", 96, (3, 3), padding="SAME"), Conv2D_bn("conv5-b-3", 96, (3, 3), strides=(2,2))],
        [MaxPooling2D("apool5", (3,3), strides=(2,2))]
         ,name = "concat5"
    ),
    Concat(
        [Conv2D_bn("conv6-a", 192, (1, 1))],
        [Conv2D_bn("conv6-b-1", 128, (1, 1)), Conv2D_bn("conv6-b-2", 128, (1, 7), padding="SAME"), Conv2D_bn("conv6-b-3", 192, (7, 1), padding="SAME")],
        [Conv2D_bn("conv6-c-1", 128, (1, 1)),Conv2D_bn("conv6-c-2", 128, (7, 1) , padding="SAME"),Conv2D_bn("conv6-c-3", 128, (1, 7) , padding="SAME"),Conv2D_bn("conv6-c-4", 128, (7, 1) , padding="SAME"),Conv2D_bn("conv6-c-5", 192, (1, 7) , padding="SAME")],
        [AveragePooling2D("apool6", (3,3), strides=(1,1), padding="SAME"), Conv2D_bn("conv6-d", 192, (1,1), padding="SAME")]
        ,name="concat6"
    ),
    Concat(
        [Conv2D_bn("conv7-a", 192, (1, 1))],
        [Conv2D_bn("conv7-b-1", 160, (1, 1)), Conv2D_bn("conv7-b-2", 160, (1, 7), padding="SAME"), Conv2D_bn("conv7-b-3", 192, (7, 1), padding="SAME")],
        [Conv2D_bn("conv7-c-1", 160, (1, 1)),Conv2D_bn("conv7-c-2", 160, (7, 1) , padding="SAME"),Conv2D_bn("conv7-c-3", 160, (1, 7) , padding="SAME"),Conv2D_bn("conv7-c-4", 160, (7, 1) , padding="SAME"),Conv2D_bn("conv7-c-5", 192, (1, 7) , padding="SAME")],
        [AveragePooling2D("apool7", (3,3), strides=(1,1), padding="SAME"), Conv2D_bn("conv7-d", 192, (1,1), padding="SAME")]
        ,name="concat7"
    ),
    Concat(
        [Conv2D_bn("conv8-a", 192, (1, 1))],
        [Conv2D_bn("conv8-b-1", 160, (1, 1)), Conv2D_bn("conv8-b-2", 160, (1, 7), padding="SAME"), Conv2D_bn("conv8-b-3", 192, (7, 1), padding="SAME")],
        [Conv2D_bn("conv8-c-1", 160, (1, 1)),Conv2D_bn("conv8-c-2", 160, (7, 1) , padding="SAME"),Conv2D_bn("conv8-c-3", 160, (1, 7) , padding="SAME"),Conv2D_bn("conv8-c-4", 160, (7, 1) , padding="SAME"),Conv2D_bn("conv8-c-5", 192, (1, 7) , padding="SAME")],
        [AveragePooling2D("apool8", (3,3), strides=(1,1), padding="SAME"), Conv2D_bn("conv8-d", 192, (1,1), padding="SAME")]
        ,name="concat8"
    ),
    Concat(
        [Conv2D_bn("conv9-a", 192, (1, 1))],
        [Conv2D_bn("conv9-b-1", 192, (1, 1)), Conv2D_bn("conv9-b-2", 192, (1, 7), padding="SAME"), Conv2D_bn("conv9-b-3", 192, (7, 1), padding="SAME")],
        [Conv2D_bn("conv9-c-1", 192, (1, 1)),Conv2D_bn("conv9-c-2", 192, (7, 1) , padding="SAME"),Conv2D_bn("conv9-c-3", 192, (1, 7) , padding="SAME"),Conv2D_bn("conv9-c-4", 192, (7, 1) , padding="SAME"),Conv2D_bn("conv9-c-5", 192, (1, 7) , padding="SAME")],
        [AveragePooling2D("apool9", (3,3), strides=(1,1), padding="SAME"), Conv2D_bn("conv9-d", 192, (1,1), padding="SAME")]
        ,name="concat9"
    ),
    Concat(
        [Conv2D_bn("conva-a", 192, (1, 1)), Conv2D_bn("conva-a2", 320, (3, 3), strides=(2, 2))],
        [Conv2D_bn("conva-b-1", 192, (1, 1)),
         Conv2D_bn("conva-b-2", 192, (1, 7), padding="SAME"),
         Conv2D_bn("conva-b-3", 192, (7, 1), padding="SAME"),
         Conv2D_bn("conva-b-4", 192, (3, 3), strides=(2,2))],
        [MaxPooling2D("poola")],
        name="concata"
    ),
    Concat(
        [Conv2D_bn("convb-a", 320, (1, 1))],
        [Concat([Conv2D_bn("convb-b-1", 384, (1, 1))],
                [Conv2D_bn("convb-b-2", 384, (1, 3), padding="SAME"),
                 Conv2D_bn("convb-b-3", 384, (3, 1), padding="SAME")],
                name="concatb-1")],
        [Concat([Conv2D_bn("convb-c-1", 448, (1, 1)),
                 Conv2D_bn("convb-c-2", 384, (3, 3), padding="SAME")],
                [Conv2D_bn("convb-c-3", 384, (1, 3), padding="SAME"),
                 Conv2D_bn("convb-c-4", 384, (3, 1), padding="SAME")],
                name="concatb-2")],
        [AveragePooling2D("poolb", (3,3), strides=(1,1) , padding="SAME"),
         Conv2D_bn("convb-d", 192, (1, 1), padding="SAME")]
        ,name="concatb"
    ),
    Concat(
        [Conv2D_bn("convc-a", 320, (1, 1))],
        [Concat([Conv2D_bn("convc-b-1", 384, (1, 1))],
                [Conv2D_bn("convc-b-2", 384, (1, 3), padding="SAME"),
                 Conv2D_bn("convc-b-3", 384, (3, 1), padding="SAME")],
                name="concatc-1")],
        [Concat([Conv2D_bn("convc-c-1", 448, (1, 1)),
                 Conv2D_bn("convc-c-2", 384, (3, 3), padding="SAME")],
                [Conv2D_bn("convc-c-3", 384, (1, 3), padding="SAME"),
                 Conv2D_bn("convc-c-4", 384, (3, 1), padding="SAME")],
                name="concatc-2")],
        [AveragePooling2D("poolc", (3,3), strides=(1,1) , padding="SAME"),
         Conv2D_bn("convc-d", 192, (1, 1), padding="SAME")]
        ,name="concatc"
    ),
    GlobalAveragePooling2D("gpoold"),
    Dropout(),
    FullConnect("fc", NUM_CLASSES, activationProc=tf.sigmoid)
]

WSCALE=math.sqrt(2.0/NUM_CLASSES)
modelFile = "model_celeba.ckpt"
scoreFileName = "score_celeba.ckpt"
num_gpu = 2

# 画像サイズを既に加工済みなら、cropが速い
imageResize = "crop"
dataType = "multi-label"
faceType = "real"

trainFile = "train.txt"
testFile = "test.txt"
validFile = "not exists"

isLargeDataset = True
