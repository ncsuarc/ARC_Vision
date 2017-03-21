import numpy as np
import cv2
import ARC
import os

from collections import deque

from PyQt5.QtCore import (Qt, QObject, QRunnable, QThreadPool, QTimer, pyqtSignal, QAbstractListModel, QModelIndex)
from PyQt5.QtWidgets import (QWidget, QListView, QPushButton, QScrollArea, QApplication, QHBoxLayout, QVBoxLayout, QFileDialog, QLabel)

from ui_utils import *
import filters

class MainWindow(QWidget):
    def __init__(self, parent=None, flight_number=0, threads=4):
        super(MainWindow, self).__init__(parent)

        self.flight_number = flight_number
        self.threads = threads

        self.flight = ARC.Flight(flight_number)

        self.initUI()
        
        self.lq_listener = ARC.db.Listener(self.flight.database, ARC.dbschema.notify_image_low_quality)
        self.hq_listener = ARC.db.Listener(self.flight.database, ARC.dbschema.notify_image_high_quality)

        self.queryNewImagesTimer = QTimer()
        self.queryNewImagesTimer.timeout.connect(self.queryNewImages)
        self.queryNewImagesTimer.start(500)

        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(threads)

        self.images = deque(self.flight.all_images()) 
        self.queuedImages = {}

        self.queueCount = 0

    def initUI(self):
        self.roiDisplayScroll = QScrollArea(self)
        self.roiDisplayScroll.setWidgetResizable(True)
        self.roiDisplay = QWidget()
        self.roiLayout = QVBoxLayout()
        self.roiDisplay.setLayout(self.roiLayout)
        self.roiDisplayScroll.setWidget(self.roiDisplay)

        self.targetDisplayScroll = QScrollArea(self)
        self.targetDisplayScroll.setWidgetResizable(True)
        self.targetDisplay = QWidget()
        self.targetLayout = QVBoxLayout()
        self.targetDisplay.setLayout(self.targetLayout)
        self.targetDisplayScroll.setWidget(self.targetDisplay)

        self.imageryLayout = QVBoxLayout()
        self.imageryLayout.addWidget(self.targetDisplayScroll)
        self.imageryLayout.addWidget(self.roiDisplayScroll)

        self.infoDisplay = QWidget()
        self.infoDisplay.setMinimumWidth(200)

        self.waitingList = QListView()
        self.waitingListModel = StringListModel(self.waitingList)
        self.waitingList.setModel(self.waitingListModel)

        self.processingList = QListView()
        self.processingListModel = StringListModel(self.processingList)
        self.processingList.setModel(self.processingListModel)

        self.finishedList = QListView()
        self.finishedListModel = StringListModel(self.finishedList)
        self.finishedList.setModel(self.finishedListModel)

        self.infoLayout = QVBoxLayout()
        self.infoLayout.addWidget(QLabel("Waiting for Processing:"))
        self.infoLayout.addWidget(self.waitingList)
        self.infoLayout.addWidget(QLabel("Currently Processing:"))
        self.infoLayout.addWidget(self.processingList)
        self.infoLayout.addWidget(QLabel("Completed:"))
        self.infoLayout.addWidget(self.finishedList)

        self.infoDisplay.setLayout(self.infoLayout)

        hbox = QHBoxLayout(self)
        hbox.addLayout(self.imageryLayout)
        hbox.addWidget(self.infoDisplay)

        self.setLayout(hbox)

    def queryNewImages(self):
        try:
            lq_id = int(self.lq_listener.next(timeout=0.05))
            #TODO check low quality image for target
            #If likely target identified, request high quality
        except StopIteration:
            pass

        try:
            hq_id = int(self.hq_listener.next(timeout=0.05))
            self.images.append(self.flight.image(hq_id))
        except StopIteration:
            pass

        self.processImages()

    def processImages(self):
        if self.queueCount < self.threads*2 and len(self.images) > 0:
            while not self.images[0].nadired:
                self.images.popleft() #Throw out images that are not nadired
            self.startImageProcessing(self.images.popleft())

    def startImageProcessing(self, image):
        self.waitingListModel.addItem(image.high_quality_jpg)
        self.queuedImages[image.image_id] = image.high_quality_jpg
        processor = ImageProcessor(image,
                lambda: self.processingStarted(image.image_id),
                lambda: self.processingFinished(image.image_id),
                self.newTarget)
        self.pool.start(processor)
        self.queueCount += 1

    def processingStarted(self, image_id):
        self.waitingListModel.removeItem(self.queuedImages[image_id])
        self.processingListModel.addItem(self.queuedImages[image_id])

    def processingFinished(self, image_id):
        self.processingListModel.removeItem(self.queuedImages[image_id])
        self.finishedListModel.addItem(self.queuedImages[image_id])
        self.queueCount -= 1

    def newTarget(self, target_image):
        for i in range(self.roiLayout.count()): 
            print(self.roiLayout.itemAt(i).widget().roi.distance(target_image))
        self.roiLayout.addWidget(ROICanvas(target_image))

    def keyPressEvent(self, evt):
        super(MainWindow, self).keyPressEvent(evt)
    
    def closeEvent(self, event):
        saveDirectory = str(QFileDialog.getExistingDirectory(self, "Select a directory to save the output in..."))

        try:
            if not os.path.isdir(saveDirectory + "/targets"):
                os.mkdir(saveDirectory + "/targets")
            if not os.path.isdir(saveDirectory + "/fp"):
                os.mkdir(saveDirectory + "/fp")

            t_n = 0
            fp_n = 0

            for i in range(self.roiLayout.count()): 
                local_widget = self.roiLayout.itemAt(i).widget()
                if(local_widget.target):
                    local_widget.saveRoiImage(saveDirectory + "/targets/t{}.jpg".format(t_n))
                    t_n += 1
                else:
                    local_widget.saveRoiImage(saveDirectory + "/fp/f{}.jpg".format(fp_n))
                    fp_n += 1
        except Exception as e:
            print('While saving images, the following exception occurred:')
            print(e)

        if not self.pool.waitForDone():
            print('Processing killed before completion.')

class StringListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super(StringListModel, self).__init__()
        self._strings = []
        self._removed_strings = []

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if index.row() > len(self._strings):
            return None

        if role == Qt.DisplayRole:
            return self._strings[index.row()]

        return None

    def flags(self, index):
        flags = super(StringListModel, self).flags(index)

        return flags

    def insertRows(self, row, count, parent=QModelIndex()):
        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        self._strings[row:row] = [''] * count
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent=QModelIndex()):
        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
        del self._strings[row:row + count]
        self.endRemoveRows()
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._strings)

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False

        self._strings[index.row()] = value
        self.dataChanged.emit(index, index)
        return True
    
    def addItem(self, string):
        if string in self._removed_strings:
            self._removed_strings.remove(string)
            return
        index = self.rowCount()
        self.insertRows(index, 1)
        self.setData(self.index(index), string)

    def removeItem(self, string):
        try:
            return self.removeRows(self._strings.index(string), 1)
        except ValueError:
            self._removed_strings.append(string)
            return False

class ImageProcessorConnector(QObject):

    started = pyqtSignal()
    finished = pyqtSignal()
    new_target = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        super(ImageProcessorConnector, self).__init__()

class ImageProcessor(QRunnable): 
    def __init__(self, image, started_callback, finished_callback, new_target_callback):
        super(ImageProcessor, self).__init__()
        
        self.setAutoDelete(True)

        self.image = image
        self._emitter = ImageProcessorConnector()
        self._emitter.started.connect(started_callback)
        self._emitter.finished.connect(finished_callback)
        self._emitter.new_target.connect(new_target_callback)

    def run(self):
        try:
            self._emitter.started.emit()
            rois = filters.get_targets(self.image)
            for roi in rois:
                self._emitter.new_target.emit(roi)
            self._emitter.finished.emit()
        except Exception as e:
            print(e)

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
