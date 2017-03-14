import cv2
import tensorflow as tf
import numpy as np

import roi_model

import argparse
from os import listdir
from random import shuffle

parser = argparse.ArgumentParser(description='Search images for targets.')
parser.add_argument("-i", "--input", required=True, help="Directory to search")
args = vars(parser.parse_args())

images = []
labels = []
files = listdir(args['input'] + "/fp")
for f in files:
    img = cv2.imread(args['input'] + "/fp/"+f)
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())
    labels.append(0)

files = listdir(args['input'] + "/targets")
for f in files:
    img = cv2.imread(args['input'] + "/targets/"+f)
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
    cnn_model = roi_model.Model(sess, load=True, n_classes=2)
    cnn_model.train(sess, images, labels)
