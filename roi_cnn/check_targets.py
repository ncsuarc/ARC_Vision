from .roi_model import Model
from .tfsession import TFSession

import numpy as np
import cv2

sess = TFSession()
print("Loading False Positive model...")
cnn_model = Model(sess, load=True)

def check_targets(images):
    labels = cnn_model.test(sess, [cv2.resize(image, (60, 60)).flatten() for image in images])
    return [bool(np.argmax(label)) for label in labels]
