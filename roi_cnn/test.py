import cv2
import tensorflow as tf
import numpy as np
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

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    predicted_labels = cnn_model.test(sess, images)
    for (img, pred) in zip(images, predicted_labels):
        idx = np.argmax(pred)
        print(pred)
        if(bool(idx)):
            cv2.imshow("Display", img.reshape((60,60,3)))
            cv2.waitKey()
