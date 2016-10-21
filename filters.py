import cv2
import numpy as np

def filter_primary(image):
    imgs = []
    
    mask = color_mask(image)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 12)

    canny = cv2.Canny(mask, 100, 200)
    im2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if (cv2.contourArea(cnt) > 100) and (cv2.contourArea(cnt) < 4000):
            x, y, width, height = cv2.boundingRect(cnt)
            roi = image[y:y+height, x:x+width]
            imgs.append(roi)
    return imgs

def check_target(image): 
    kernel = np.ones((3,3), np.uint8)
    mask = color_mask(image)
    mask = cv2.erode(mask, kernel, iterations = 1)
    
    res = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    if h == 0:
        return False
    ar = (float(w)/h)
    m = cv2.moments(mask)
    if(cv2.contourArea(np.array([(x,y), (x,y+h), (x+w,y+h), (x+w,y)])) > 20 ) and (ar > 0.65 and ar < 1.54) and (m['m00'] > 5000):
        return True
    return False

def color_mask(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(image_hsv, np.array([0, 125, 150]), np.array([360, 150, 255]))
    blue_mask = cv2.inRange(image_hsv, np.array([150, 100, 100]), np.array([200, 230, 230]))
    yellow_mask = cv2.inRange(image_hsv, np.array([0, 125, 125]), np.array([40, 255, 255]))
    green_mask = cv2.inRange(image_hsv, np.array([60, 50, 100]), np.array([120, 100, 255]))
    
    mask = cv2.bitwise_or(red_mask, blue_mask)
    mask = cv2.bitwise_or(mask, yellow_mask)    
    mask = cv2.bitwise_or(mask, green_mask)
    return mask
