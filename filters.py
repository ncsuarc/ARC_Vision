import cv2
import numpy as np
import math
import ROI
from roi_cnn.check_targets import check_targets

def high_pass_filter(arc_image):
    filename = arc_image.filename
    image = cv2.imread(filename[:-3] + 'jpg')

    ROIs = []
    
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(image_blur, 75, 150)

    (_, contours, _) = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#    cnt_out = np.zeros(image.shape, np.uint8)
    for cnt in contours:
        try:
            roi = ROI.ROI(arc_image, image, cnt)
            ROIs.append(roi)
#            cv2.drawContours(cnt_out, [roi.hull], 0, (255,255,255), 3)
        except ValueError as e:
            print("Not a target: " + str(e))
            continue
   
#    images = [roi.roi for roi in ROIs] 
#    if len(images) == 0:
#        return []
#    labels = check_targets(images)
#    for roi, label, img in zip(ROIs, labels, images):
#        cv2.imshow('other', img);
#        print(label)
#        cv2.waitKey()
#        if(label):
#            cv2.drawContours(cnt_out, [roi.hull], 0, (255,255,255), 3)

#    dst = cv2.addWeighted(image,0.5,cnt_out,0.5,0)
#    cv2.imshow('Display', dst)
#    cv2.waitKey()
    
    return ROIs
