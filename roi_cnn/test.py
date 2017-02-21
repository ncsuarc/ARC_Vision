import cv2
import tensorflow as tf
import numpy as np
import roi_model

import argparse
from os import listdir, path

parser = argparse.ArgumentParser(description='Search images for targets.')
parser.add_argument("-i", "--input", required=True, help="Directory to search")
args = vars(parser.parse_args())

images = []

for f in listdir(args['input']):
    img = cv2.imread(path.join(args['input'], f))
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())

# Launch the graph
with tf.Session() as sess:
    cnn_model = roi_model.Model(sess)
    predicted_labels = cnn_model.test(sess, images)
    for (img, pred) in zip(images, predicted_labels):
        idx = np.argmax(pred)
        if(bool(idx)):
            cv2.imshow("Display", img.reshape((60,60,3)))
            cv2.waitKey()
