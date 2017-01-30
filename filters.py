import cv2
import numpy as np
import math
import ROI
from roi_cnn.check_targets import check_targets

def high_pass_filter(arc_image):
    try:
        canny_low
        canny_high
    except NameError:
        global canny_low
        global canny_high
        canny_low = 100
        canny_high = 250

    filename = arc_image.filename
    image = cv2.imread(filename[:-3] + 'jpg')

    ROIs = []
    
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    goal = 300
    while True:
        canny = cv2.Canny(image_blur, canny_low, canny_high)

        (_, contours, _) = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        step = (goal-n)/5
        print("Contours:{}  Low:{} High:{} Step:{}".format(n, canny_low, canny_high, step))
        if n > (goal*4/3):
            canny_low += step
            if canny_low < 10:
                canny_high -= step
                canny_low = canny_high*0.7
        elif n < (goal*2/3):
            canny_low += step
            if canny_low >= canny_high:
                canny_high -= step
                canny_low = canny_high*0.7
        else:
            break

    print('Filtering')

    cnt_out = np.zeros(image.shape, np.uint8)
    for cnt in contours:
        try:
            roi = ROI.ROI(arc_image, image, cnt)
            ROIs.append(roi)
            cv2.drawContours(cnt_out, [roi.hull], 0, (255, 0, 0), 3)
        except ValueError as e:
            continue
   
    images = [roi.roi for roi in ROIs] 
    if len(images) == 0:
        return []
    labels = check_targets(images)
    for roi, label, img in zip(ROIs, labels, images):
        if(label):
            cv2.drawContours(cnt_out, [roi.hull], 0, (255, 255, 255), 3)

    canny_color = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(image,0.75,cnt_out,0.25,0)
    cv2.imshow('Display', dst)
    cv2.waitKey()
    
    return ROIs
