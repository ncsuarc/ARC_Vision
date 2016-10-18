import cv2
import numpy as np

def filter_1(image):
    imgs = []
    #Color Threshold
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    red_mask = cv2.inRange(image_hsv, np.array([0, 125, 150]), np.array([360, 150, 255]))
    blue_mask = cv2.inRange(image_hsv, np.array([150, 100, 100]), np.array([200, 230, 230]))
    yellow_mask = cv2.inRange(image_hsv, np.array([0, 125, 125]), np.array([40, 255, 255]))
    green_mask = cv2.inRange(image_hsv, np.array([60, 50, 100]), np.array([120, 100, 255]))
    
    mask = cv2.bitwise_or(red_mask, blue_mask)
    mask = cv2.bitwise_or(mask, yellow_mask)    
    mask = cv2.bitwise_or(mask, green_mask)    
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 12)
    res = cv2.bitwise_and(image, image, mask=mask)

    canny = cv2.Canny(mask, 100, 200)
    im2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
#    contour_draw = np.zeros(image.shape)
    for cnt in contours:
        if (cv2.contourArea(cnt) > 100) and (cv2.contourArea(cnt) < 4000):
            x, y, width, height = cv2.boundingRect(cnt)
            roi = image[y:y+height, x:x+width]
            resize_dimensions = (500, int(height * (500/width)))
            roi = cv2.resize(roi, resize_dimensions, cv2.INTER_NEAREST)
            imgs.append(roi)
#            cv2.drawContours(contour_draw, [cnt], 0, (255, 255, 255), -1)
#
#    mask = cv2.inRange(contour_draw, np.array([100, 100, 100]), np.array([255, 255, 255]))
#    res = cv2.bitwise_and(image, image, mask=mask) 
#    res = cv2.addWeighted(res,0.8,image,0.2,0)
#    return [res]
    return imgs
