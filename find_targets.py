import numpy as np
import cv2
import ARC
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QLabel)
from PyQt5.QtGui import (QImage, QPainter, QColor)

import filters

class MainWindow(QWidget):
    def __init__(self, parent=None, flight_number=0):
        super(MainWindow, self).__init__(parent)

        self.imageCanvas = ROICanvas(parent=self, flight_number=flight_number)
        self.initUI()

    def initUI(self):
        nextButton = QPushButton("Next")
        prevButton = QPushButton("Previous")
        
        nextButton.clicked.connect(self.imageCanvas.nextImage)
        prevButton.clicked.connect(self.imageCanvas.prevImage)
        
        hbox = QHBoxLayout(self)
        hbox.addStretch(1)
        hbox.addWidget(self.imageCanvas)
        
        vbox = QVBoxLayout()
        vbox.addWidget(nextButton)
        vbox.addWidget(prevButton)
        
        hbox.addLayout(vbox)
        self.setLayout(hbox)

    def keyPressEvent(self, QKeyEvent):
        super(MainWindow, self).keyPressEvent(QKeyEvent)

class ROICanvas(QWidget):
    
    def __init__(self, parent=None, flight_number=0):
        super(ROICanvas, self).__init__(parent)
        
        #self.label = QLabel("Regions of Interest", parent=self)
        self.mQImages = []
        self.ROIs = []

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
            
            self.n = -1 #Next image will iterate this to 0
            self.nextImage()
        except ValueError as e:
            print(e)

    def nextImage(self): 
        self.n += 1
        if self.n >= len(self.images):
            return
        self.show_image_ROIs()
        #if len(self.mQImages) == 0:
        #    self.nextImage()

    def prevImage(self):
        self.n -= 1
        if self.n <= 0:
            return
        self.show_image_ROIs()
        #if len(self.mQImages) == 0:
        #    self.prevImage()

    def show_image_ROIs(self):
        del self.mQImages[:]
        self.mQImages = []
        self.ROIs = filters.high_pass_filter(self.images[self.n])
        #ROIs = filters.false_positive_filter(ROIs)
        for roi in self.ROIs:
            image = cv2.resize(roi.roi, (60,60))
            self.mQImages.append(cvImgToQImg(image))
        self.parent().update()

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)

        painter.drawImage(10, 10, cvImgToQImg(self.ROIs[0].image))

        x_off = 10
        y_off = (self.geometry().height()/2) + 10
        for img in self.mQImages:
            painter.drawImage(x_off, y_off, img)
            x_off += 70
            if x_off >= self.geometry().width():
                x_off = 10
                y_off += 70
        painter.end()

def cvImgToQImg(cvImg):
    qimg = QImage(cvImg.shape[1], cvImg.shape[0], QImage.Format_RGB888)
    for x in range(cvImg.shape[1]-1):
        for y in range(cvImg.shape[0]-2):
            qimg.setPixel(x, y, QColor(*(cvImg[x,y])).rgb())
    return qimg

if __name__=="__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Search flight images for targets.')
    parser.add_argument("-i", "--input_flight", help="Flight number to search")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    if args.input_flight:
        w = MainWindow(flight_number=args.input_flight)
    else:
        w = MainWindow()
    w.resize(1600, 900)
    w.show()
    app.exec_()
