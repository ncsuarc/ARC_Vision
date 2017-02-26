import cv2

from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QWidget 
from PyQt5.QtGui import (QImage, QPainter, QColor)

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
        self.setMaximumHeight(70)
        self.setMaximumWidth(70)
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
