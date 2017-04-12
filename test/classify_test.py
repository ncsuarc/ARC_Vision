import unittest

import ARC

import cv2
import numpy as np

import classify 
import roi

class ClassifyTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_check_targets(self):
        target = cv2.imread('test_images/target.jpg')
        false_positive = cv2.imread('test_images/fp.jpg')
        labels = classify.check_targets([target, false_positive])
        self.assertTrue(labels[0])
        self.assertFalse(labels[1])

if __name__ == '__main__':
    unittest.main()
