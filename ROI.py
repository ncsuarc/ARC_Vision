import cv2
import numpy as np
import filters
class ROI():
    def __init__(self, arc_image, image, cnt):
        self.arc_image = arc_image
        self.image = image
        
        x, y, w, h = cv2.boundingRect(cnt)
        self.roi = self.image[y:y+h, x:x+w]
        
        rect = _, _, angle = cv2.minAreaRect(cnt)

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bound_rect = self.roi
        cv2.drawContours(bound_rect, [box], 0,(0,255,0),2)
#        cv2.imshow('Unrotated', bound_rect)
#        cv2.waitKey()

        (c_x, c_y) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        full_width = int((h * sin) + (w * cos))
        full_height = int((h * cos) + (w * sin))
        M[0, 2] += (full_width / 2) - c_x
        M[1, 2] += (full_height / 2) - c_y

        self.roi = cv2.warpAffine(image, M, (full_width, full_height))
        if not self.validate():
            raise ValueError("Failed validation test.")
    def validate(self):
        gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(gray, 20, 100)
        im2, contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
#        cv2.imshow('Rotated', edged)
#        cv2.waitKey()
        if len(contours) < 1:
            return False
        return True
