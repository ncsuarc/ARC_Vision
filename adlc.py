import numpy as np
import ARC

from PyQt5.QtCore import(QObject, QRunnable, QThreadPool, QTimer, pyqtSignal)
from PyQt5.QtWidgets import (QWidget, QPushButton, QScrollArea, QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog, QLabel)
from PyQt5.QtGui import (QImage, QPainter, QColor)

from ui_utils import *
import filters

class MainWindow(QWidget):
    def __init__(self, parent=None, flight_number=0, threads=6):
        super(MainWindow, self).__init__(parent)

        self.flight_number = flight_number
        
        self.flight = ARC.Flight(flight_number)

        self.initUI()
        
        self.lq_listener = ARC.db.Listener(self.flight.database, ARC.dbschema.notify_image_low_quality)
        self.hq_listener = ARC.db.Listener(self.flight.database, ARC.dbschema.notify_image_high_quality)

        self.queryNewImagesTimer = QTimer()
        self.queryNewImagesTimer.timeout.connect(self.queryNewImages)
        self.queryNewImagesTimer.start(1000)

        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(threads)

    def initUI(self):
        self.targetDisplayScroll = QScrollArea(self)
        self.targetDisplayScroll.setWidgetResizable(True)
        self.targetDisplay = QWidget()
        self.targetLayout = QGridLayout()
        self.targetDisplay.setLayout(self.targetLayout)
        self.targetDisplayScroll.setWidget(self.targetDisplay)

        self.infoDisplay = QWidget()
        self.infoDisplay.setMinimumWidth(400)

        self.infoLayout = QVBoxLayout()
        self.infoLayout.addWidget(QLabel("Currently Processing:"))
        self.infoLayout.addStretch(1)

        self.infoDisplay.setLayout(self.infoLayout)

        hbox = QHBoxLayout(self)
        hbox.addWidget(self.targetDisplayScroll)
        hbox.addWidget(self.infoDisplay)

        self.setLayout(hbox)

    def queryNewImages(self):
        try:
            lq_id = int(self.lq_listener.next(timeout=0.1))
            #TODO check low quality image for target
            #If likely target identified, request high quality
        except StopIteration:
            pass

        try:
            hq_id = int(self.hq_listener.next(timeout=0.1))
            self.processImage(self.flight.image(hq_id))
        except StopIteration:
            pass

    def processImage(self, image):
        processor = ImageProcessor(image, lambda: self.processingFinished(image.image_id))
        self.pool.start(processor)

    def processingFinished(self, image_id):
        print(image_id)

    def newTarget(self, target_image):
        print(target_image)

    def keyPressEvent(self, evt):
        super(MainWindow, self).keyPressEvent(evt)

class ImageProcessorConnector(QObject):

    finished = pyqtSignal()
    new_target = pyqtSignal(np.ndarray)

    def __init__(self):
        super(ImageProcessorConnector, self).__init__()

class ImageProcessor(QRunnable): 

    def __init__(self, image, finished_callback):
        super(ImageProcessor, self).__init__()
        self.image = image
        self._emitter = ImageProcessorConnector()
        self._emitter.finished.connect(finished_callback)

    def run(self):
        rois = filters.get_targets(self.image)
        for roi in rois:
            self._emitter.new_target.emit(roi)
        self._emitter.finished.emit()

if __name__=="__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Search flight images for targets.')
    parser.add_argument("-i", "--input_flight", help="Flight number to search")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    w = MainWindow(flight_number=args.input_flight)
    w.resize(1600, 900)
    w.show()
    app.exec_()
