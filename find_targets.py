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
        hbox.addWidget(self.imageCanvas)
        
        vbox = QVBoxLayout()
        vbox.addWidget(nextButton)
        vbox.addWidget(prevButton)
        vbox.addStretch(1)

        hbox.addLayout(vbox)
        self.setLayout(hbox)

    def keyPressEvent(self, evt):
        super(MainWindow, self).keyPressEvent(evt)

class ROICanvas(QWidget):
    def __init__(self, parent=None, flight_number=0):
        super(ROICanvas, self).__init__(parent)
        
        self.mQImages = []
        self.ROIs = []
        self.images = []
        self.currentImage = None
        self.roi_height = 0

        try:
            flight = ARC.Flight(flight_number)
            targets = flight.all_targets()

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
        self.currentImage = cv2.imread(self.images[self.n].filename[:-3]+'jpg')

        for roi in self.ROIs:
            image = cv2.resize(roi.roi, (60,60))
            self.mQImages.append(cvImgToQImg(image))
        self.parent().repaint()

    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)

        new_width = self.geometry().width()
        new_height = int(new_width*self.currentImage.shape[0]/self.currentImage.shape[1])
        painter.drawImage(0, 0, cvImgToQImg(cv2.resize(self.currentImage, (new_width, new_height))))

        x_off = 0
        y_off = new_height + 10
        self.roi_height = y_off
    
        for img, roi in zip(self.mQImages, self.ROIs):
            if not roi.target:
                painter.drawImage(x_off, y_off, img)
            x_off += 70
            if x_off + 70 >= self.geometry().width():
                x_off = 0
                y_off += 70
        painter.end()

    def mousePressEvent(self, evt):
        super(ROICanvas, self).mousePressEvent(evt)
        y = int((evt.y()-self.roi_height)/70)
        if y < 0:
            return
        idx = (y * int(self.geometry().width()/70)) + int(evt.x()/70)
        try:
            self.ROIs[idx].target = not (self.ROIs[idx].target)
            self.repaint()
        except:
            return
def cvImgToQImg(cvImg):
    return QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], cvImg.strides[0], QImage.Format_RGB888)

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
