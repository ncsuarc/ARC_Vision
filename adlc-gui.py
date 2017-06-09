import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from PyQt5.QtWidgets import (QMainWindow, QWidget, QListView, QPushButton, QScrollArea, QApplication, QHBoxLayout, QVBoxLayout, QFileDialog, QLabel, QAction)

from ui_utils import *
from adlc import ADLCProcessor


class MainWindow(QMainWindow):
    def __init__(self, flight_number=0, threads=4, check_interop=True):
        super(MainWindow, self).__init__(None)

        self.initUI()
        self.initToolbar()

        self.processor = ADLCProcessor(flight_number = flight_number, check_interop=check_interop)
        self.processor.new_target.connect(self.new_target)
        self.processor.new_roi.connect(self.new_roi)

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

        self.imageryLayout = QHBoxLayout()
        self.imageryLayout.addWidget(self.targetDisplayScroll)
        self.imageryLayout.addWidget(self.roiDisplayScroll)

        centralWidget = QWidget()
        centralWidget.setLayout(self.imageryLayout)

        self.setCentralWidget(centralWidget)

    def initToolbar(self):
        saveAction = QAction('Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save Images')
        saveAction.triggered.connect(self.saveImages)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveAction)

    def new_target(self, target):
        self.targetLayout.addWidget(TargetCanvas(target))

    def new_roi(self, roi):
        self.roiLayout.addWidget(ROICanvas(roi))

    def saveImages(self):
        saveDirectory = str(QFileDialog.getExistingDirectory(self, "Select a directory to save the output in..."))

        try:
            if not os.path.isdir(saveDirectory + "/targets"):
                os.mkdir(saveDirectory + "/targets")
            if not os.path.isdir(saveDirectory + "/fp"):
                os.mkdir(saveDirectory + "/fp")
            if not os.path.isdir(saveDirectory + "/thumbnail"):
                os.mkdir(saveDirectory + "/thumbnail")
            if not os.path.isdir(saveDirectory + "/thumbnail/targets"):
                os.mkdir(saveDirectory + "/thumbnail/targets")
            if not os.path.isdir(saveDirectory + "/thumbnail/fp"):
                os.mkdir(saveDirectory + "/thumbnail/fp")

            t_n = 0
            fp_n = 0

            for i in range(self.roiLayout.count()): 
                local_widget = self.roiLayout.itemAt(i).widget()
                if(local_widget.target):
                    local_widget.saveRoiImage(saveDirectory + "/targets/t{}.jpg".format(t_n))
                    local_widget.saveThumbnailImage(saveDirectory + "/thumbnail/targets/t{}.jpg".format(t_n))
                    t_n += 1
                else:
                    local_widget.saveRoiImage(saveDirectory + "/fp/f{}.jpg".format(fp_n))
                    local_widget.saveThumbnailImage(saveDirectory + "/thumbnail/fp/f{}.jpg".format(fp_n))
                    fp_n += 1
        except Exception as e:
            print('While saving images, the following exception occurred:')
            print(e)

if __name__=="__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Search flight images for targets.')
    parser.add_argument("-i", "--input_flight", help="Flight number to search")
    parser.add_argument("--no-interop", action="store_true")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    w = MainWindow(flight_number=args.input_flight, check_interop=(not args.no_interop))
    w.resize(1600, 900)
    w.show()
    app.exec_()
