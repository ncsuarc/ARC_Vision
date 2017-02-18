import unittest

import ARC
import filters

class FilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.flight = ARC.Flight(157)
        self.targets = self.flight.all_targets()
        self.target_images = []
        
        for tgt in self.targets:
            if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
                continue
            new_images = self.flight.images_near(tgt.coord, 30)
            self.target_images.extend(new_images)
        #Remove duplicate files
        self.target_images = dict((image.filename, image) for image in self.target_images).values()
        self.images = flight.all_images()
        
    def test_high_pass_filter(self):
        for img in self.target_images:
            ROIs = filters.high_pass_filter(img, goal=1000)
            self.assertLess(len(ROIs), 1200)
            self.assertGreater(len(ROIs), 800)

if __name__ == '__main__':
    unittest.main()
