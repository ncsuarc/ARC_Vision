import cv2
import tensorflow as tf

import roi_model

from os import listdir
from os.path import isfile, join

cnn_model = roi_model.Model()

init = tf.global_variables_initializer()
images = []

for i in range(0,1214):
    img = cv2.imread("roi/roi_{}.jpg".format(i))
    img = cv2.resize(img, (60,60))
    images.append(img.flatten())

labels = [0] * len(images)

for line in open("roi/labels.txt", 'r'):
    labels[int(line)] = 1

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    cnn_model.train(sess, images, labels)
