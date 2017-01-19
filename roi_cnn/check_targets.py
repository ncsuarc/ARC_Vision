from .roi_model import Model
import tensorflow as tf
import numpy as np
import cv2

sess = tf.Session()
cnn_model = Model(sess)
def check_targets(images):
    labels = cnn_model.test(sess, [cv2.resize(image, (60, 60)).flatten() for image in images])
    return [bool(np.argmax(label)) for label in labels]
