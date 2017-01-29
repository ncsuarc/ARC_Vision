import cv2
import tensorflow as tf
import numpy as np

import roi_model

from os import listdir
from random import shuffle

images = []
labels = []
targets = []
target_idxs = [int(line) for line in open("roi/labels.txt", 'r')]
for f in listdir("roi"):
    if not f.endswith(".jpg"):
        continue
    img = cv2.imread("roi/"+f)
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())
    print(f[3:-4])
    if int(f[3:-4]) in target_idxs:
        labels.append(1)
        targets.append(img)
    else:
        labels.append(0)

#images = np.array(images)
#labels = np.array(labels)
#rng_state = np.random.get_state()
#np.random.shuffle(images)
#np.random.set_state(rng_state)
#np.random.shuffle(labels)

# Launch the graph
with tf.Session() as sess:
    cnn_model = roi_model.Model(sess, False)
    cnn_model.train(sess, images, labels)
