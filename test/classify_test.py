import unittest
import os

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

    def test_classify_shape(self):
        test_dir = 'test_images/shapes'
        i = 0
        for directory in os.listdir(test_dir):
            if(os.path.isdir(os.path.join(test_dir, directory))):
                for image_file in os.listdir(os.path.join(test_dir, directory)):
                    with self.subTest(i=i):
                        i += 1
                        image = cv2.imread(os.path.join(test_dir, directory, image_file))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.assertEqual(classify.classify_shape(image)[0][1], directory)


if __name__ == '__main__':
    unittest.main()
