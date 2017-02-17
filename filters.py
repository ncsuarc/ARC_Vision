import cv2
import numpy as np
import math
import ROI
from roi_cnn.check_targets import check_targets

def high_pass_filter(arc_image, goal=600):
    try:
        num_iter
        num_images
        avg_first
    except NameError:
        global num_iter
        global num_images
        global avg_first
        num_iter = 0
        num_images = 0
        avg_first = 0.0
    num_images += 1
    filename = arc_image.filename
    image = cv2.imread(filename[:-3] + 'jpg')

    ROIs = []
    
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    canny_low = 100
    canny_high = 325
    error = 0
    error_total = 0
    error_prev = 0
    first = True
    for i in range(20):
        num_iter += 1
        canny = cv2.Canny(image_blur, canny_low, canny_high)

        (_, contours, _) = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        
        if first:
            avg_first += n
            error_prev = goal - n
            first = False
        else:
            error_prev = error

        error = goal - n
        error_total += error
        step = 0.03635*error + 0.052*error_total + 0.004*(error-error_prev)
        if step > 50:
            step = 50
        elif step < -50:
            step = -50
        if abs(error) < 15:
            break
        else:
            canny_high -= step
            if canny_low >= canny_high:
                canny_low -= step

    for cnt in contours:
        try:
            roi = ROI.ROI(arc_image, image, cnt)
            ROIs.append(roi)
        except ValueError as e:
            continue
   
    return ROIs

def false_positive_filter(old_ROIs):
    if len(old_ROIs) == 0:
        return []

    new_ROIs = []
    images = [roi.roi for roi in old_ROIs] 
    labels = check_targets(images)
    for roi, label in zip(old_ROIs, labels):
        if(label):
            new_ROIs.append(roi)
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
