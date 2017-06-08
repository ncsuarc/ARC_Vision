import unittest

import roi

class FilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_haversine(self):
        pt1 = (36.16, -78.08)
        pt2 = (36.1689935, -78.08)

        self.assertTrue(approx_equal(roi.haversine(pt1, pt2), 1000, 1))

def approx_equal(a, b, tol):
    return abs(a - b) < tol
if __name__=='__main__':
    unittest.main()
