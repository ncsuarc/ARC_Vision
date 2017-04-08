import ARC.nn

import numpy as np
import cv2

class Classifier:
    class __Classifier:
        def __init__(self):
            print("Loading False Positive model...")
            self.fp_model = ARC.nn.Model('models/false_positive')
            print("Loading Shape model...")
            self.shape_model = ARC.nn.Model('models/shape')
            print("Loading Alphanumeric model...")
            self.alphanumeric_model = ARC.nn.Model('models/alphanumeric')

    instance = None

    def __init__(self):
        if not Classifier.instance:
            Classifier.instance = Classifier.__Classifier()

    def __getattr__(self, name):
        return getattr(self.instance, name)

def check_targets(images):
    labels = Classifier().fp_model.test([cv2.resize(image, (64, 64)).flatten() for image in images])
    return [bool(np.argmax(label)) for label in labels]
