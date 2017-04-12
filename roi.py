import cv2
import numpy as np
from math import radians, cos, sin, asin, sqrt
import filters
import classify

class Target():

    MIN_DISTANCE = 100
    MIN_MATCHES = 4
    MATCH_THRESHOLD = 0.70

    bf = cv2.BFMatcher()

    def __init__(self, roi):
        self.rois = []
        self.nadired_rois = []
        self.alphanumerics = {}
        self.shapes = {}

        self._total_lat = 0
        self._total_lon = 0
        self.add_roi(roi)

    def get_confidence(self):
        return len(self.rois)

    def get_shape(self):
        return sorted(list(self.shapes.items()), key = lambda x: x[1], reverse = True)[0][0]

    def get_alphanumeric(self):
        return sorted(list(self.alphanumerics.items()), key = lambda x: x[1], reverse = True)[0][0]

    def add_roi(self, roi):
        roi.classify()

        self.rois.append(roi)
        
        if len(self.rois) == 1:
            self.coord = roi.coord

        if(roi.arc_image.nadired):
            self.nadired_rois.append(roi)
            lat, lon = roi.coord
            self._total_lat += lat
            self._total_lon += lon
            self.coord = (self._total_lat / len(self.nadired_rois), self._total_lon / len(self.nadired_rois))

        #Update target information based on the new roi's properties
        for label in roi.shape_labels:
            if label[1] in self.shapes.keys():
                self.shapes[label[1]] += label[0]
            else:
                self.shapes[label[1]] = label[0]

        for label in roi.alphanumeric_labels:
            if label[1] in self.alphanumerics.keys():
                self.alphanumerics[label[1]] += label[0]
            else:
                self.alphanumerics[label[1]] = label[0]

    def is_duplicate(self, other):
        if haversine(self.coord, other.coord) > Target.MIN_DISTANCE:
            return False
        for roi in self.rois:
            matches = Target.bf.knnMatch(roi.descriptor, other.descriptor, k=2)

            good = 0
            try: 
                for m,n in matches:
                    if m.distance < Target.MATCH_THRESHOLD * n.distance:
                        good += 1
            except ValueError:
                continue
            
            if good < Target.MIN_MATCHES:
                continue

            self.add_roi(other)
            return True
        return False

class ROI():

    MIN_WIDTH = 0.5
    MAX_WIDTH = 2.0
    MIN_HEIGHT = 0.5
    MAX_HEIGHT = 2.0
    THUMBNAIL_BORDER = 0.25

    sift = cv2.xfeatures2d.SIFT_create()

    def __init__(self, arc_image, image, cnt):
        self.arc_image = arc_image

        #Check height and width
        x, y, w, h = cv2.boundingRect(cnt)
        self.real_width = w*arc_image.width_m_per_px
        self.real_height = h*arc_image.height_m_per_px
        
        if not ((ROI.MIN_WIDTH <= self.real_width <= ROI.MAX_WIDTH) and (ROI.MIN_HEIGHT <= self.real_height <= ROI.MAX_HEIGHT)):
            raise ValueError("Failed size test.")
        
        roi_mask = np.zeros(image.shape[0:2], np.uint8)
        cv2.drawContours(roi_mask, [cnt], 0, 255, -1)

        #Adjust coordinates to include a quarter meter more on each side
        diff_m_width = int(ROI.THUMBNAIL_BORDER / arc_image.width_m_per_px)
        diff_m_height = int(ROI.THUMBNAIL_BORDER / arc_image.height_m_per_px)
        if(y - diff_m_height < 0):
            diff_m_height = y
        if(x - diff_m_width < 0):
            diff_m_width = x
        if(y + h + diff_m_height > image.shape[0]):
            diff_m_height = image.shape[0] - y - h
        if(x + w + diff_m_width > image.shape[1]):
            diff_m_width = image.shape[1] - x - w

        self.thumbnail = image[y - diff_m_height : y + h + diff_m_height, x - diff_m_width : x + w + diff_m_width]
        roi_mask = roi_mask[y - diff_m_height : y + h + diff_m_height, x - diff_m_width : x + w + diff_m_width]

        #Find the target contour
        contours = filters.get_contours(roi_mask, goal=1)
        
        if(len(contours) == 0):
            raise ValueError("Failed contour test.")

        self.cnt = contours[np.argmax([cv2.contourArea(c) for c in contours])]
        self.rect = cv2.minAreaRect(self.cnt)

        if not self.validate():
            raise ValueError("Failed validation test.")
        
        #This will rescale the image from 0-255 to 128-255
        #As a result, black targets will have color values as 128, as opposed to 0
        sub_image = rescale_img_values(self.thumbnail)
        #Now, when the mask is applied, black values are at 128, and the background is 0
        self.roi = cv2.bitwise_and(sub_image, sub_image, mask=roi_mask)
        
        M = cv2.moments(self.cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        self.coord = arc_image.coord(x = cX, y = cY)

        ##Create a nice thumbnail image for this ROI

        try:
            self.keypoints, self.descriptor = ROI.sift.detectAndCompute(self.thumbnail, None)
            if self.descriptor == None:
                raise ValueError('Unable to detect keypoints')
        except Exception:
            raise ValueError('Unable to detect keypoints')

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

    def classify(self):
        try:
            ((self.shape_mask, self.shape_color), (self.alphanumeric_mask, self.alphanumeric_color)) = filters.get_target_info(self.roi)
        except IndexError:
            raise ValueError("Error identifying target shape and letter")

        self.shape_labels = classify.classify_shape(filters.draw_mask_color(self.shape_mask, self.shape_color))
        self.alphanumeric_labels = classify.classify_alphanumeric(filters.draw_mask_color(self.alphanumeric_mask, self.alphanumeric_color))

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
    Returns: the distance between the two points in meters.
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [pt1[1], pt1[0], pt2[1], pt2[0]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    m = 6371008 * c
    return m

#Renormalize values
def rescale_img_values(img):
    temp_img = (img.astype(float) / 2) + 128
    return temp_img.astype(np.uint8)

def descale_img_values(img):
    temp_img = (img.astype(float) - 128)
    temp_img[np.where((temp_img < [0, 0, 0]).all(axis = 2))] = [0, 0, 0]
    temp_img *= 2
    return temp_img.astype(np.uint8)

def descale_color_value(color):
    color = np.array(list(color))
    temp_color = (color.astype(float) - 128)
    temp_color[temp_color < 0] = 0
    temp_color *= 2
    return tuple(temp_color.astype(np.uint8).tolist())
