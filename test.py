import unittest
import time

import ARC
import filters
import cv2
import numpy as np
import roi

class FilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Display', 1920, 1080)

        self.flight = ARC.Flight(165)
        self.targets = self.flight.all_targets()
        self.target_images = []
        if self.targets == None:
            self.targets = []

        for tgt in self.targets:
            if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
                continue
            new_images = self.flight.images_near(tgt.coord, 50)
            self.target_images.extend(new_images)
        #Remove duplicate files
        self.target_images = dict((image.filename, image) for image in self.target_images).values()
        self.images = self.flight.all_images()
        
    def test_get_contours(self):
        for img, i in zip(self.images, range(len(self.target_images))):
            with self.subTest(i=i):
                start_time = time.time()
                contours = filters.get_contours(cv2.imread(img.high_quality_jpg), goal=300)
                self.assertLess((time.time()-start_time), 2) # Ensure the operation took less than 2 seconds
                self.assertLess(len(contours), 330)
                self.assertGreater(len(contours), 270)

    def test_get_contours_small(self):
        redblue = cv2.imread('test_images/redblue.png')
        contours = filters.get_contours(redblue, goal=1)
        self.assertEqual(len(contours), 1)

    def test_get_contours_canny(self):
        for img in self.images:
            image = cv2.imread(img.high_quality_jpg)
            (contours, canny) = filters.get_contours(image, goal=300, getCanny=True)
            canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

            res = cv2.addWeighted(image, 0.6, canny, 0.4, 0)
            cv2.imshow('Display', res)
            cv2.waitKey()

    def test_get_rois(self):
        for img, i in zip(self.images, range(len(self.target_images))):
            rois = filters.get_rois(img)
            for roi in rois:
                cv2.imshow('Display', roi.roi)
                cv2.waitKey()

    def test_get_targets(self):
        for img in self.target_images:
            rois = filters.get_targets(img)
            for roi in rois:
                cv2.imshow('Display', roi.roi)
                cv2.waitKey()
                cv2.imshow('Display', roi.roi_original)
                cv2.waitKey()
    
    def test_get_target_info(self):
        #Test a red target with a blue letter
        redblue = cv2.imread('test_images/redblue.png')
        redblue = cv2.cvtColor(redblue, cv2.COLOR_BGR2RGB)
        
        shape, letter = filters.get_target_info(redblue)
        shape_color = roi.descale_color_value(shape[1])

        self.assertLessEqual(shape_color[0], 255)
        self.assertGreaterEqual(shape_color[0], 250)
        self.assertLessEqual(shape_color[1], 5)
        self.assertGreaterEqual(shape_color[1], 0)
        self.assertLessEqual(shape_color[2], 5)
        self.assertGreaterEqual(shape_color[2], 0)
        
        letter_color = roi.descale_color_value(letter[1])

        self.assertLessEqual(letter_color[0], 5)
        self.assertGreaterEqual(letter_color[0], 0)
        self.assertLessEqual(letter_color[1], 5)
        self.assertGreaterEqual(letter_color[1], 0)
        self.assertLessEqual(letter_color[2], 255)
        self.assertGreaterEqual(letter_color[2], 250)
        
        #Test a black target with a blue letter
        blackblue = cv2.imread('test_images/blackblue.png')
        blackblue = cv2.cvtColor(blackblue, cv2.COLOR_BGR2RGB)
        
        shape, letter = filters.get_target_info(blackblue)
        shape_color = roi.descale_color_value(shape[1])

        self.assertLessEqual(shape_color[0], 5)
        self.assertGreaterEqual(shape_color[0], 0)
        self.assertLessEqual(shape_color[1], 5)
        self.assertGreaterEqual(shape_color[1], 0)
        self.assertLessEqual(shape_color[2], 5)
        self.assertGreaterEqual(shape_color[2], 0)
        
        letter_color = roi.descale_color_value(letter[1])

        self.assertLessEqual(letter_color[0], 5)
        self.assertGreaterEqual(letter_color[0], 0)
        self.assertLessEqual(letter_color[1], 5)
        self.assertGreaterEqual(letter_color[1], 0)
        self.assertLessEqual(letter_color[2], 255)
        self.assertGreaterEqual(letter_color[2], 250)
        
        #Test a red target with a black letter
        redblack = cv2.imread('test_images/redblack.png')
        redblack = cv2.cvtColor(redblack, cv2.COLOR_BGR2RGB)
        
        shape, letter = filters.get_target_info(redblack)
        shape_color = roi.descale_color_value(shape[1])

        self.assertLessEqual(shape_color[0], 255)
        self.assertGreaterEqual(shape_color[0], 250)
        self.assertLessEqual(shape_color[1], 5)
        self.assertGreaterEqual(shape_color[1], 0)
        self.assertLessEqual(shape_color[2], 5)
        self.assertGreaterEqual(shape_color[2], 0)
        
        letter_color = roi.descale_color_value(letter[1])

        self.assertLessEqual(letter_color[0], 5)
        self.assertGreaterEqual(letter_color[0], 0)
        self.assertLessEqual(letter_color[1], 5)
        self.assertGreaterEqual(letter_color[1], 0)
        self.assertLessEqual(letter_color[2], 5)
        self.assertGreaterEqual(letter_color[2], 0)
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(FilterTest("test_get_target_info"))
    suite.addTest(FilterTest("test_get_contours_small"))
    suite.addTest(FilterTest("test_get_contours"))
    #suite.addTest(FilterTest("test_get_contours"))
    #suite.addTest(FilterTest("test_get_rois"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
