import cv2
import numpy as np
import math
import roi
from classify import check_targets

def get_targets(arcImage):
    return false_positive_filter(get_rois(arcImage))

def get_contours(image, goal, getCanny=False, P=0.05):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    canny_low = 20
    canny_high = 200
    
    for i in range(10):
        canny = cv2.Canny(image_blur, canny_low, canny_high)

        (_, contours, _) = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        
        error = goal - n
        step = P * error
        
        step = coerceVar(step, -50, 50)
        
        if abs(error) < math.ceil(0.1*goal): #10% margin of error
            break
        else:
            canny_high -= step
            canny_high = coerceVar(canny_high, 0, 500)
            if canny_low >= canny_high:
                canny_low -= step
                canny_low = coerceVar(canny_low, 0, canny_high)
    if getCanny:
        return (contours, canny)
    return contours

def coerceVar(var, minimum, maximum):
    if var < minimum:
        return minimum
    elif var > maximum:
        return maximum
    else:
        return var

def get_rois(arc_image, goal = 600, min_size = 0.25, max_size = 2):
    image = cv2.imread(arc_image.high_quality_jpg)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rois = []
    for cnt in get_contours(image, goal):
        try:
            region = roi.ROI(arc_image, image, cnt)
            rois.append(region)
        except ValueError as e:
            continue
   
    return rois

def false_positive_filter(old_ROIs):
    if len(old_ROIs) == 0:
        return []

    new_ROIs = []
    images = [region.thumbnail for region in old_ROIs] 
    labels = check_targets(images)
    for region, label in zip(old_ROIs, labels):
        if(label):
            new_ROIs.append(region)
    return new_ROIs

