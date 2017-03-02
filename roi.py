import cv2
import numpy as np
import math
import filters

class ROI():
    def __init__(self, arc_image, image, cnt):
        hull = cv2.convexHull(cnt)

        x, y, w, h = cv2.boundingRect(hull)
        real_width = w*arc_image.width_m_per_px
        real_height = h*arc_image.height_m_per_px
        if not ((0.25 <= real_width <= 2.0) and (0.25 <= real_height <= 2.0)):
            raise ValueError("Failed size test.")
            
        rect = cv2.minAreaRect(hull)

        roi_mask = np.zeros(image.shape[0:2], np.uint8)
        cv2.drawContours(roi_mask, [hull], 0, 255, -1)
        image_masked = cv2.bitwise_and(image, image, mask=roi_mask)
        
        self.roi = image_masked[y:y+h, x:x+w]
        self.rect = rect
        self.hull = hull
        self.arc_image = arc_image
        self.image = image
        if not self.validate():
            raise ValueError("Failed validation test.")
    
    def validate(self):
        #check area of the hull compared to the area of the rect
        hull_area = cv2.contourArea(self.hull)
        rect_cnt = cv2.boxPoints(self.rect)
        rect_cnt = np.int0(rect_cnt)
        rect_area = cv2.contourArea(rect_cnt)

        if (rect_area*.5) > hull_area:
            return False
        #Calculate aspect ratio of rotated bounding box
        tl, tr, br, bl = self.order_points(rect_cnt)
        if self.dist(tl, bl) == 0:
            ar = 0
        else:
            ar = self.dist(tl, tr)/self.dist(tl, bl) 
        if not (0.3 < ar < 3):
            return False
        
        return True

    def order_points(self, pts):
        s = pts.sum(axis = 1)
        diff = np.diff(pts, axis = 1)
        tl = pts[np.argmin(s)]
        tr = pts[np.argmin(diff)]
        br = pts[np.argmax(s)]
        bl = pts[np.argmax(diff)]
        
        return (tl, tr, br, bl)
        
    def dist(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
