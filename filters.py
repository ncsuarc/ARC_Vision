import cv2
import numpy as np
import math

def filter_primary(image):
    imgs = []
    
    mask = color_mask(image)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 12)

    canny = cv2.Canny(mask, 100, 200)
    im2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt_out = np.zeros(image.shape)
    cnt_out = np.uint8(cnt_out)
    for cnt in contours:
        if (cv2.contourArea(cnt) > 100) and (cv2.contourArea(cnt) < 4000):
            x, y, width, height = cv2.boundingRect(cnt)
            roi = image[y:y+height, x:x+width]
            imgs.append(roi)
            cv2.drawContours(cnt_out, [cnt], 0, (255,255,255), 3)
    
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(image,0.7,mask_rgb,0.3,0)
    #cv2.imshow('Display', dst)
    #cv2.waitKey()
    
    return imgs

def check_target(image):
    kernel = np.ones((3,3), np.uint8)
    mask = color_mask(image)
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    
    m = cv2.moments(mask)

    #Check that there is a substantial number of pixels present, rather than a handful of interesting specks of grass
    if m['m00'] > 1500:
        #Run KMeans with K=3 for grass, target, and character
        Z = image.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS, 10, 0.25)
        compactness,labels,centers = cv2.kmeans(Z,4,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS ) 

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        res = res.reshape((image.shape))
         
        mask = color_mask(res)
        mask = cv2.erode(mask, kernel, iterations = 1)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        
        pts = cv2.findNonZero(mask)
        if pts == None:
            return False
        rect = cv2.minAreaRect(pts)
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        #Calculate aspect ratio of rotated bounding box
        tl, tr, _, bl = order_points(box)
        if dist(tl, bl) == 0:
            ar = 0
        else:
            ar = dist(tl, tr)/dist(tl, bl)        
        
#        cv2.imshow('Display', res)
#        if(cv2.waitKey() == 115):
#            cv2.imwrite("image.jpg", image)
        print('\t' + str(ar))
        print('\t' + str(compactness))
        if (ar > 0.3) and (ar < 3): # and (compactness > 900000) and (compactness < 2100000):
            return True
    return False

def color_mask(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(image_hsv, np.array([0, 64, 200]), np.array([10, 255, 255]))
    yellow_mask = cv2.inRange(image_hsv, np.array([10, 128, 200]), np.array([30, 255, 255]))
    green_mask1 = cv2.inRange(image_hsv, np.array([30, 50, 100]), np.array([60, 100, 255]))
    green_mask2 = cv2.inRange(image_hsv, np.array([60, 50, 100]), np.array([120, 100, 255]))
    blue_mask = cv2.inRange(image_hsv, np.array([100, 100, 100]), np.array([160, 255, 255]))
    red_mask2 = cv2.inRange(image_hsv, np.array([170, 100, 100]), np.array([180, 180, 255]))
   
    mask = cv2.bitwise_or(red_mask1, red_mask2)
    mask = cv2.bitwise_or(mask, blue_mask)
    mask = cv2.bitwise_or(mask, yellow_mask)    
    mask = cv2.bitwise_or(mask, green_mask1) 
    mask = cv2.bitwise_or(mask, green_mask2)

    return mask

def order_points(pts):
    s = pts.sum(axis = 1)
    diff = np.diff(pts, axis = 1)
    tl = pts[np.argmin(s)]
    tr = pts[np.argmin(diff)]
    br = pts[np.argmax(s)]
    bl = pts[np.argmax(diff)]

    return (tl, tr, br, bl)

def dist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
