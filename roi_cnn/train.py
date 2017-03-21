import cv2
import tensorflow as tf
import numpy as np

import model

import argparse
from os import listdir
from random import shuffle

parser = argparse.ArgumentParser(description='Search images for targets.')
parser.add_argument("-i", "--input", required=True, help="Directory to search")
parser.add_argument("-s", "--save-location", required=True, help="Directory to save and load model in.")
args = vars(parser.parse_args())

images = []
labels = []

for f in listdir(args['input'] + "/fp"):
    if not (f.endswith('jpg') or f.endswith('png')):
        continue
    img = cv2.imread(args['input'] + "/fp/"+f)
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())
    labels.append(0)

for f in listdir(args['input'] + "/targets"):
    if not (f.endswith('jpg') or f.endswith('png')):
        continue
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
cnn_model = model.Model(args['save_location'])
cnn_model.train(images, labels)
