import cv2
import numpy as np
import filters
class ROI():
    def __init__(self, arc_image, image, cnt):
        hull = cv2.convexHull(cnt)

        x, y, w, h = cv2.boundingRect(hull)
        rect = _, _, angle = cv2.minAreaRect(hull)

        roi_mask = np.zeros(image.shape[0:2], np.uint8)
        cv2.drawContours(roi_mask, [hull], 0, 255, -1)
        image_masked = cv2.bitwise_and(image, image, mask=roi_mask)
        self.roi = image_masked[y:y+h, x:x+w]
        
        (c_x, c_y) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        full_width = int((h * sin) + (w * cos))
        full_height = int((h * cos) + (w * sin))
        M[0, 2] += (full_width / 2) - c_x
        M[1, 2] += (full_height / 2) - c_y
        
        self.rect = rect
        self.hull = hull
        self.angle = angle
        self.arc_image = arc_image
        self.image = image
        self.roi = cv2.warpAffine(self.roi, M, (full_width, full_height))
        if not self.validate():
            raise ValueError("Failed validation test.")

    def validate(self):
        #check area of the hull compared to the area of the rect
        hull_area = cv2.contourArea(self.hull)
        rect_cnt = cv2.boxPoints(self.rect)
        rect_cnt = np.int0(rect_cnt)
        rect_area = cv2.contourArea(rect_cnt)

        if (rect_area*.7) > hull_area:
            return False
        return True
