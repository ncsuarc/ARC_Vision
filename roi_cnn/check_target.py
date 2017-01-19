from .roi_model import Model
import tensorflow as tf
import numpy as np
import cv2

sess = tf.Session()
cnn_model = Model(sess)
def check_target(image):
    image = cv2.resize(image, (60,60))
    sess.run(tf.global_variables_initializer())
    labels = cnn_model.test(sess, [image.flatten()])
    return bool(np.argmax(labels[0]))
