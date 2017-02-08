import numpy as np
import cv2
import ARC
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import (QWidget, QPushButton, QScrollArea, QApplication, QHBoxLayout, QVBoxLayout, QGridLayout)
from PyQt5.QtGui import (QImage, QPainter, QColor)

import filters

class MainWindow(QWidget):
    def __init__(self, parent=None, flight_number=0):
        super(MainWindow, self).__init__(parent)

        self.flight_number = flight_number
        self.images = []
        self.ROI_canvases = []

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
        self.roiDisplayScroll.setWidgetResizable(True)
        self.roiDisplayScroll.setMinimumWidth(400)
        self.roiDisplayScroll.setMaximumWidth(400)
        self.roiDisplay = QWidget()
        self.roiLayout = QGridLayout()
        self.roiDisplay.setLayout(self.roiLayout)
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

        self.setLayout(hbox)

    def nextImage(self): 
        self.n += 1
        if self.n >= len(self.images):
            return
        self.update_images()

    def prevImage(self):
        self.n -= 1
        if self.n <= 0:
            return
        self.update_images()

    def update_images(self):
        for i in reversed(range(self.roiLayout.count())): 

            self.roiLayout.itemAt(i).widget().setParent(None)

        ROIs = filters.high_pass_filter(self.images[self.n])
        #ROIs = filters.false_positive_filter(ROIs)
        x = 0
        y = 0
        for roi in ROIs:
            self.roiLayout.addWidget(roi_canvas, y, x)
            x += 1
            if x > 2:
                x = 0
                y += 1
            
        self.imageCanvas.setImage(cv2.imread(self.images[self.n].filename[:-3] + 'lowquality.jpg'))
        self.update()

    def keyPressEvent(self, evt):
        super(MainWindow, self).keyPressEvent(evt)

class ImageCanvas(QWidget):
    def __init__(self, parent=None, image=None):
        super(ImageCanvas, self).__init__(parent)
        self.setImage(image)

    def setImage(self, image):
        try:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass

    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)

        new_width = self.geometry().width()
        new_height = int(new_width*self.image.shape[0]/self.image.shape[1])
        painter.drawImage(0, 0, cvImgToQImg(cv2.resize(self.image, (new_width, new_height))))

class ROICanvas(ImageCanvas):
    def __init__(self, roi):
        super().__init__(image=roi.roi)
        self.target = False
        self.roi = roi
        self.setMinimumHeight(70)
        self.setMinimumWidth(70)
        self.setMaximumHeight(self.roi.roi.shape[0] if self.roi.roi.shape[0] >= 70 else 70)
        self.setMaximumWidth(self.roi.roi.shape[1] if self.roi.roi.shape[1] >= 70 else 70)
    def mousePressEvent(self, evt):
        super(ROICanvas, self).mousePressEvent(evt)
        self.target = not self.target
        self.repaint()

    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)

        if self.target:
            painter.setBrush(QColor(0, 255, 0))
            painter.drawRect(QRect(-1, -1, self.geometry().width()+2, self.geometry().height()+2))
        painter.drawImage(5, 5, cvImgToQImg(cv2.resize(self.image, (60, 60))))

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
