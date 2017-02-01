import cv2
import numpy as np
import math
import ROI
from roi_cnn.check_targets import check_targets

def high_pass_filter(arc_image, goal=500):
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

    canny_low = 125
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
        #print("\tContours:{}  Low:{} High:{} Step:{}".format(n, canny_low, canny_high, step))
        if abs(error) < 15:
            break
        #elif abs(error) < 30:
        #    canny_low += step
        else:
            canny_high -= step
            if canny_low >= canny_high:
                canny_low -= step

    print("{} iterations/image, {} initial".format(num_iter/num_images, avg_first/num_images))

    cnt_out = np.zeros(image.shape, np.uint8)
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
