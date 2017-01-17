import cv2
import numpy as np
import math
import ROI

def high_pass_filter(arc_image):
    filename = arc_image.filename
    image = cv2.imread(filename[:-3] + 'jpg')
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    ROIs = []

    image_blur = cv2.GaussianBlur(image, (3, 3), 0)
    canny = cv2.Canny(image_blur, 75, 150)

    (_, contours, _) = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt_out = np.zeros(image.shape, np.uint8)
    for cnt in contours:
        try:
            _, _, width, height = cv2.boundingRect(cnt)
            real_width = width*arc_image.width_m_per_px
            real_height = height*arc_image.height_m_per_px
            if (0.5 <= real_width <= 2) and (0.5 <= real_height <= 2):
                roi = ROI.ROI(arc_image, image, cnt)
                ROIs.append(roi)
                cv2.drawContours(cnt_out, [cnt], 0, (255,255,255), 3)
        except Exception as e:
            print("Not a target: " + str(e))
            continue
    
    dst = cv2.addWeighted(image,0.5,cnt_out,0.5,0)
    cv2.imshow('Display', dst)
    cv2.waitKey()
    
    return ROIs
