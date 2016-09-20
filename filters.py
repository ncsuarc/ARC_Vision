import cv2
import numpy as np

def filter_1(image):
    imgs = []
    #Otsu adaptive threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #Erode and dialate to remove small noise 
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 6)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    res = cv2.bitwise_and(image, image, mask=mask)

    #RGB Threshold
    mask = cv2.inRange(res, np.array([0,0,0]), np.array([80, 255, 150]))
    mask = cv2.bitwise_not(mask, mask)

    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 8)
    res = cv2.bitwise_and(image, image, mask=mask)

    canny = cv2.Canny(mask, 100, 200)
    im2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contour_draw = np.zeros(image.shape)
    for cnt in contours:
        if (cv2.contourArea(cnt) > 100) and (cv2.contourArea(cnt) < 2000):
            x, y, width, height = cv2.boundingRect(cnt)
            roi = image[y:y+height, x:x+width]
            resize_dimensions = (500, int(height * (500/width)))
            roi = cv2.resize(roi, resize_dimensions, cv2.INTER_NEAREST)
            #imgs.append(roi)
            cv2.drawContours(contour_draw, [cnt], 0, (255, 255, 255), -1)
    
    mask = cv2.inRange(contour_draw, np.array([100, 100, 100]), np.array([255, 255, 255]))
    res = cv2.bitwise_and(image, image, mask=mask) 
    
    #return imgs
    return [res]
