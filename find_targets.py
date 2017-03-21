import numpy as np
import cv2
import ARC
import os

from PyQt5.QtWidgets import (QWidget, QPushButton, QScrollArea, QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog)
from PyQt5.QtGui import (QImage, QPainter, QColor)

from ui_utils import *
import filters

class MainWindow(QWidget):
    def __init__(self, parent=None, flight_number=0):
        super(MainWindow, self).__init__(parent)

        self.saveDirectory = str(QFileDialog.getExistingDirectory(self, "Select a directory to save the output in..."))

        self.flight_number = flight_number
        self.images = []

        self.t_n = 0
        self.fp_n = 0

        self.image_flagged = False

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
        nextButton.clicked.connect(self.nextImage)

        self.flagButton = QPushButton("Flag Image")
        self.flagButton.setCheckable(True)
        self.flagButton.clicked[bool].connect(self.flagImage)

        ctl = QHBoxLayout()
        ctl.addWidget(self.flagButton)
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
            QApplication.quit()
            return
        self.update_images()

    def flagImage(self, flagged):
        self.image_flagged = not self.image_flagged

    def update_images(self):
        if not os.path.isdir(self.saveDirectory + "/targets"):
            os.mkdir(self.saveDirectory + "/targets")
        if not os.path.isdir(self.saveDirectory + "/fp"):
            os.mkdir(self.saveDirectory + "/fp")
        for i in reversed(range(self.roiLayout.count())): 
            local_widget = self.roiLayout.itemAt(i).widget()
            if(local_widget.target):
                local_widget.saveRoiImage(self.saveDirectory + "/targets/t{}.jpg".format(self.t_n))
                self.t_n += 1
            else:
                local_widget.saveRoiImage(self.saveDirectory + "/fp/f{}.jpg".format(self.fp_n))
                self.fp_n += 1
                
            self.roiLayout.itemAt(i).widget().setParent(None)

        if self.image_flagged:
            with open(self.saveDirectory + "/flagged.txt", "a") as flaggedFile:
                flaggedFile.write(self.images[self.n-1].filename + '\n')

        self.flagButton.setChecked(False)
        self.image_flagged = False

        ROIs = filters.high_pass_filter(self.images[self.n], goal=600)
        target_ROIs = filters.false_positive_filter(ROIs)
        x = 0
        y = 0
        for roi in ROIs:
            new_roi_canvas = ROICanvas(roi)
            if roi in target_ROIs:
                new_roi_canvas.target=True
            self.roiLayout.addWidget(ROICanvas(roi), y, x)
            x += 1
            if x > 2:
                x = 0
                y += 1
            
        self.imageCanvas.setImage(cv2.imread(self.images[self.n].filename[:-3] + 'lowquality.jpg'))
        self.update()

    def keyPressEvent(self, evt):
        super(MainWindow, self).keyPressEvent(evt)

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
