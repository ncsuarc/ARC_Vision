import cv2
import numpy as np
from math import radians, cos, sin, asin, sqrt
import filters

class ROI():
    def __init__(self, arc_image, image, cnt):
        self.arc_image = arc_image

        #Check height and width
        x, y, w, h = cv2.boundingRect(cnt)
        self.real_width = w*arc_image.width_m_per_px
        self.real_height = h*arc_image.height_m_per_px

        if not ((0.5 <= self.real_width <= 2.0) and (0.5 <= self.real_height <= 2.0)):
            raise ValueError("Failed size test.")

        roi_mask = np.zeros(image.shape[0:2], np.uint8)
        cv2.drawContours(roi_mask, [cnt], 0, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.dilate(roi_mask, kernel, iterations = 1)
        roi_mask = cv2.erode(roi_mask, kernel, iterations = 2)
        roi_mask = cv2.dilate(roi_mask, kernel, iterations = 1)
        
        (_, contours, _) = cv2.findContours(roi_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        if(len(contours) == 0):
            raise ValueError("Failed contour test.")

        self.cnt = contours[0]
        self.rect = cv2.minAreaRect(self.cnt)

        if not self.validate():
            raise ValueError("Failed validation test.")
        M = cv2.moments(self.cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        self.coord = arc_image.coord(x = cX, y = cY)

        image_masked = cv2.bitwise_and(image, image, mask=roi_mask)
        
        self.roi = image_masked[y:y+h, x:x+w]

        #Adjust coordinates to include half a meter more on each side
        diff_m_width = int(0.25 / arc_image.width_m_per_px)
        diff_m_height = int(0.25 / arc_image.height_m_per_px)
        if(y - diff_m_height < 0):
            diff_m_height = y
        if(x - diff_m_width < 0):
            diff_m_width = x
        if(y + h + diff_m_height > image.shape[0]):
            diff_m_height = image.shape[0] - y - h
        if(x + w + diff_m_width > image.shape[1]):
            diff_m_width = image.shape[1] - x - w

        self.roi_original = image[y - diff_m_height : y + h + diff_m_height, x - diff_m_width : x + w + diff_m_width]
    
    def validate(self):
        #check area of the contour compared to the area of the rect
        cnt_area = cv2.contourArea(self.cnt)
        rect_cnt = cv2.boxPoints(self.rect)
        rect_cnt = np.int0(rect_cnt)
        rect_area = cv2.contourArea(rect_cnt)

        if (rect_area*.5) > cnt_area:
            return False
        #Calculate aspect ratio of rotated bounding box
        tl, tr, br, bl = order_points(rect_cnt)
        if cartesian_dist(tl, bl) == 0:
            self.ar = 0
        else:
            self.ar = cartesian_dist(tl, tr)/cartesian_dist(tl, bl) 
        if not (0.3 < self.ar < 3):
            return False
        
        return True

    def distance(self, other):
        return haversine(self.coord, other.coord)

def order_points(pts):
    s = pts.sum(axis = 1)
    diff = np.diff(pts, axis = 1)
    tl = pts[np.argmin(s)]
    tr = pts[np.argmin(diff)]
    br = pts[np.argmax(s)]
    bl = pts[np.argmax(diff)]
    
    return (tl, tr, br, bl)
    
def cartesian_dist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def haversine(pt1, pt2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [pt1[1], pt1[0], pt2[1], pt2[0]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371008 * c
    return km
