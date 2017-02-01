import numpy as np
import cv2
import ARC
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget, QPushButton, QScrollArea, QApplication, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import (QImage, QPainter, QColor)

import filters

class MainWindow(QWidget):
    def __init__(self, parent=None, flight_number=0):
        super(MainWindow, self).__init__(parent)

        self.flight_number = flight_number
        self.images = []

        self.initUI()
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

    def initUI(self):
        self.imageCanvas = ImageCanvas()

        self.roiDisplayScroll = QScrollArea(self)
        self.roiDisplay = QWidget()
        self.roiDisplayScroll.setWidget(self.roiDisplay)

        nextButton = QPushButton("Next")
        prevButton = QPushButton("Previous")
        
        nextButton.clicked.connect(self.nextImage)
        prevButton.clicked.connect(self.prevImage)
        
        ctl = QHBoxLayout()
        ctl.addWidget(prevButton)
        ctl.addStretch(1)
        ctl.addWidget(nextButton)
        
        vbox = QVBoxLayout()
        vbox.addLayout(ctl)
        vbox.addWidget(self.imageCanvas)

        hbox = QHBoxLayout(self)
        hbox.addLayout(vbox)
        hbox.addWidget(self.roiDisplayScroll)

        self.setLayout(vbox)

    def nextImage(self): 
        self.n += 1
        if self.n >= len(self.images):
            return
        self.imageCanvas.image = cv2.imread(self.images[self.n].filename[:-3] + 'jpg')
        self.update()

    def prevImage(self):
        self.n -= 1
        if self.n <= 0:
            return
        self.imageCanvas.image = self.images[self.n]
        self.update()

    def keyPressEvent(self, evt):
        super(MainWindow, self).keyPressEvent(evt)

class ImageCanvas(QWidget):
    def __init__(self, parent=None, image=None):
        super(ImageCanvas, self).__init__(parent)
        if(image):
            self.image = cv2.imread(image.filename[:-3] + 'jpg')
        
    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)

        new_width = self.geometry().width()
        new_height = int(new_width*self.image.shape[0]/self.image.shape[1])
        painter.drawImage(0, 0, cvImgToQImg(cv2.resize(self.image, (new_width, new_height))))

class ROICanvas(QWidget):
    def __init__(self, parent=None, flight_number=0):
        super(ROICanvas, self).__init__(parent)
        
        self.mQImages = []
        self.ROIs = []
        self.images = []
        self.currentImage = None
        self.roi_height = 0


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
