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
        
        for tgt in self.targets:
            if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
                continue
            new_images = self.flight.images_near(tgt.coord, 50)
            self.target_images.extend(new_images)
        #Remove duplicate files
        self.target_images = dict((image.filename, image) for image in self.target_images).values()
        self.images = self.flight.all_images()
        
    def test_get_contours(self):
        for img, i in zip(self.target_images, range(len(self.target_images))):
            with self.subTest(i=i):
                start_time = time.time()
                rois = filters.get_contours(cv2.imread(img.high_quality_jpg), goal=300)
                self.assertLess((time.time()-start_time), 2) # Ensure the operation took less than 2 seconds
                self.assertLess(len(rois), 330)
                self.assertGreater(len(rois), 270)
    
    def test_get_contours_canny(self):
        for img in self.target_images:
            image = cv2.imread(img)
            (contours, canny) = filters.get_contours(image, goal=600, getCanny=True)
            canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

            res = cv2.addWeighted(image, 0.6, canny, 0.4, 0)
            cv2.imshow('Display', res)
            cv2.waitKey()

    def test_high_pass_filter(self):
        target = self.flight.all_targets()[-1]
        images = self.flight.images_near(target.coord, 50)
        for img in images:
            rois = filters.high_pass_filter(img)
            for roi in rois:
                cv2.imshow('Display', roi.roi)
                cv2.waitKey()
        return

    def test_get_targets(self):
#        for img in self.target_images:
        target = self.flight.all_targets()[-1]
        images = self.flight.images_near(target.coord, 50)
        for img in images:
            for roi in filters.get_targets(img):
                cv2.imshow('Display', roi.roi)
                cv2.waitKey()

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(FilterTest("test_get_targets"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
