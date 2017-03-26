import cv2
import numpy as np
import math
import roi
from roi_cnn.check_targets import check_targets

def get_targets(arcImage):
    return false_positive_filter(get_rois(arcImage))

def get_contours(image, goal, getCanny=False):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    canny_low = 50
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

def get_rois(arc_image, goal=600, min_size = 0.25, max_size = 2):
    image = cv2.imread(arc_image.high_quality_jpg)

    rois = []
    contour_mask = np.zeros(image.shape[0:2], np.uint8)
    for cnt in get_contours(image, goal):
        hull = cv2.convexHull(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        real_width = w*arc_image.width_m_per_px
        real_height = h*arc_image.height_m_per_px
        if ((min_size <= real_width <= max_size) and (min_size <= real_height <= max_size)):
            cv2.drawContours(contour_mask, [hull], 0, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    contour_mask = cv2.dilate(contour_mask, kernel, iterations = 2)
    contour_mask = cv2.erode(contour_mask, kernel, iterations = 2)

    (_, contours, _) = cv2.findContours(contour_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
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

def get_target_info(img):
    #Use KMeans to segment the image into distinct colors
    #K = 3 for background, target, and letter
    K = 3
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS, 0, 0.25)
    compactness,labels,centers = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS ) 
    centers = np.uint8(centers)
    
    #Determine which class is the background
    idx = np.argmin(np.mean(centers, axis=1))
    centers[idx] = [0, 0, 0]
    labels = labels.flatten()
    
    #Determine which classes are not the background
    indexes = [i for i in range(K)]
    indexes.remove(idx)
    first_color_index = indexes[0]
    second_color_index = indexes[1]
    
    #Create two separate images, one for each non-background class
    first_labels = np.copy(labels)
    first_labels[first_labels==second_color_index] = idx
    first_img = centers[first_labels]
    first_img = first_img.reshape((img.shape))
    
    second_labels = np.copy(labels)
    second_labels[second_labels==first_color_index] = idx
    second_img = centers[second_labels]
    second_img = second_img.reshape((img.shape))
    
    #Determine shape vs letter
    (_, first_contours, first_hierarchy) = cv2.findContours(cv2.cvtColor(first_img, cv2.COLOR_RGB2GRAY),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (_, second_contours, second_hierarchy) = cv2.findContours(cv2.cvtColor(second_img, cv2.COLOR_RGB2GRAY),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if(cv2.contourArea(first_contours[0]) > cv2.contourArea(second_contours[0])):
        shape_cnt = first_contours[0]
        shape_cnts = first_contours
        shape_h = first_hierarchy
        shape_color = centers[first_color_index]
        shape_image = first_img
        
        letter_cnt = second_contours[0]
        letter_cnts = second_contours
        letter_h = second_hierarchy
        letter_color = centers[second_color_index]
        letter_img = second_img
    else:
        letter_cnt = first_contours[0]
        letter_cnts = first_contours
        letter_h = first_hierarchy
        letter_color = centers[first_color_index]
        letter_img = first_img
        
        shape_cnt = second_contours[0]
        shape_cnts = second_contours
        shape_h = second_hierarchy
        shape_color = centers[second_color_index]
        shape_image = second_img
        
    letter_mask = cv2.inRange(letter_img, letter_color, letter_color)
    
    shape_mask = np.zeros(img.shape[0:2], np.uint8)
    cv2.drawContours(shape_mask, [shape_cnt], 0, 255, -1)
    return ((shape_mask, shape_color), (letter_mask, letter_color))

def draw_mask_color(mask, color):
    img = np.zeros(mask.shape + (len(color),), np.uint8)
    
    img[:,:] = color
   
    return cv2.bitwise_and(img, img, mask=mask)
