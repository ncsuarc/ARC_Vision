import numpy as np
import cv2
import ARC
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication)
from PyQt5.QtGui import (QImage, QPainter, QColor)

import filters

class Window(QWidget):
    def __init__(self, parent=None, flight_number=0):
        super(Window, self).__init__(parent)
        self.mQImages = []

        try:
            flight = ARC.Flight(flight_number)
            targets = flight.all_targets()
            self.images = []
            
            for tgt in targets:
                if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
                    continue
                new_images = flight.images_near(tgt.coord, 50)
                self.images.extend(new_images)
            #Remove duplicate files
            self.images = list(set(self.images)) 
            
            n = 0
            while len(self.mQImages) == 0:
                self.show_image_ROIs(n)
                n += 1
        except ValueError as e:
            print(e)

    def show_image_ROIs(self, n):
        del self.mQImages[:]
        self.mQImages = []
        ROIs = filters.high_pass_filter(self.images[n])
        #ROIs = filters.false_positive_filter(ROIs)
        for roi in ROIs:
            img = QImage(roi.roi.shape[1], roi.roi.shape[0], QImage.Format_RGB888)
            for x in range(roi.roi.shape[1]):
                for y in range(roi.roi.shape[0]):
                    img.setPixel(x, y, QColor(*(roi.roi[x,y])).rgb())
            self.mQImages.append()

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        x_off = 0
        y_off = 0
        for img in self.mQImages:
            painter.drawImage(x_off, y_off, img)
            x_off += 60
            if x_off >= 1920:
                x_off = 0
                y_off += 60

        painter.end()

    def keyPressEvent(self, QKeyEvent):
        super(Window, self).keyPressEvent(QKeyEvent)

if __name__=="__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Search flight images for targets.')
    parser.add_argument("-i", "--input_flight", help="Flight number to search")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    if args.input_flight:
        w = Window(flight_number=args.input_flight)
    else:
        w = Window()
    w.resize(1920, 1080)
    w.show()
    app.exec_()
