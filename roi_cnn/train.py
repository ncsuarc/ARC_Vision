import cv2
import tensorflow as tf
import numpy as np

import roi_model

from os import listdir
from random import shuffle

images = []

for f in listdir("roi"):
    if not f.endswith(".jpg"):
        continue
    img = cv2.imread("roi/"+f)
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())

labels = [0] * len(images)

for line in open("roi/labels.txt", 'r'):
    labels[int(line)] = 1

for f in listdir("samples"):
    if not f.endswith(".png"):
        continue
    img = cv2.imread("samples/"+f)
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())
    labels.append(1)

images = np.array(images)
labels = np.array(labels)
rng_state = np.random.get_state()
np.random.shuffle(images)
np.random.set_state(rng_state)
np.random.shuffle(labels)

# Launch the graph
with tf.Session() as sess:
    cnn_model = roi_model.Model(sess, False)
    cnn_model.train(sess, images, labels)
