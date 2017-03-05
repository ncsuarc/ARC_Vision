import cv2
import numpy as np
import math
import roi
from roi_cnn.check_targets import check_targets

def get_targets(arcImage):
    return false_positive_filter(high_pass_filter(arcImage))

def get_contours(image, goal, getCanny=False):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    canny_low = 100
    canny_high = 250
    
    for i in range(20):
        canny = cv2.Canny(image_blur, canny_low, canny_high)

        (_, contours, _) = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        
        error = goal - n
        step = 0.08*error 
        
        step = coerceVar(step, -50, 50)
        
        if abs(error) < 0.1*goal: #10% margin of error
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

def high_pass_filter(arc_image, goal=600):
    image = cv2.imread(arc_image.high_quality_jpg)
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
    images = [region.roi for region in old_ROIs] 
    labels = check_targets(images)
    for region, label in zip(old_ROIs, labels):
        if(label):
            new_ROIs.append(region)
    return new_ROIs

def get_target_info(img):
    #Use KMeans to segment the image into distinct colors
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS, 0, 0.25)
    compactness,labels,centers = cv2.kmeans(Z,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS ) 
    
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res = res.reshape((img.shape))
    
    #Edge Detect
    canny = cv2.Canny(res, 100, 200)
    (_, contours, hierarchy) = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #Find the largest contour in the image, which will presumably be the target
    largest_index = 0
    largest_area = 0
    
    for c, i in zip(contours,range(len(contours))):
        area = cv2.contourArea(c)
        if area > largest_area:
            largest_index = i
            largest_area = area
    
    #Find the child of the target contour, which will be the letter
    hierarchy = hierarchy[0]
    shape_h = hierarchy[largest_index]
    shape_inner_h = hierarchy[shape_h[2]]
    letter_cnt = contours[shape_inner_h[2]]
    
    #Separate the letter and shape into different images
    letter_mask = np.zeros(img.shape[0:2], np.uint8)
    cv2.drawContours(letter_mask, [letter_cnt], 0, 255, -1)
    kernel = np.ones((3,3),np.uint8)
    letter_mask = cv2.erode(letter_mask, kernel, iterations = 1)
    letter_color = cv2.mean(img, letter_mask)
    letter_color = letter_color[0:3]
    
    shape_mask = np.zeros(img.shape[0:2], np.uint8)
    cv2.drawContours(shape_mask, contours, largest_index, 255, -1)
    shape_mask_cut = shape_mask - letter_mask
    shape_color = cv2.mean(img, shape_mask_cut)
    shape_color = shape_color[0:3]
    
    return (shape_mask, shape_color), (letter_mask, letter_color)

def draw_mask_color(mask, color):
    img = np.zeros(mask.shape.append(len(color)), np.uint8)
    
    img[:,:] = color
   
    return cv2.bitwise_and(img, img, mask=mask)
