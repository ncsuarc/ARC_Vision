import cv2
import tensorflow as tf
import numpy as np
import model

import argparse
from os import listdir, path

parser = argparse.ArgumentParser(description='Search images for targets.')
parser.add_argument("-i", "--input", required=True, help="Directory to search")
parser.add_argument("-s", "--save-location", required=True, help="Directory to save and load model in.")
args = vars(parser.parse_args())

images = []

for f in listdir(args['input']):
    if not (f.endswith('jpg') or f.endswith('png')):
        continue
    img = cv2.imread(path.join(args['input'], f))
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())

# Launch the graph
cnn_model = model.Model(args['save_location'], batch_size=10)
predicted_labels = cnn_model.test(images)
for (img, pred) in zip(images, predicted_labels):
    idx = np.argmax(pred)
    if(bool(idx)):
        cv2.imshow("Display", img.reshape((60,60,3)))
        cv2.waitKey()
