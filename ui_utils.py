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
            self.image = image
            new_width = self.geometry().width()
            new_height = int(new_width*self.image.shape[0]/self.image.shape[1])
            self.qImage = cvImgToQImg(cv2.resize(self.image, (new_width, new_height)))
        except:
            pass

    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self.qImage)

class ROICanvas(QWidget):
    def __init__(self, roi):
        super(ROICanvas, self).__init__(None)

        self.target = False
        self.roi = roi

        self.setMaximumHeight(70)
        self.setMinimumHeight(70)
        self.setMinimumWidth(70)

        self.setImage(roi.thumbnail)

    def setImage(self, image):
        new_height = self.geometry().height()-10
        new_width = int(new_height*image.shape[1]/image.shape[0])
        self.qImage = cvImgToQImg(cv2.resize(image, (new_width, new_height)))

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

        painter.drawImage(5, 5, self.qImage)
    
    def saveRoiImage(self, location):
        cv2.imwrite(location, self.roi.roi)

    def saveThumbnailImage(self, location):
        cv2.imwrite(location, self.roi.thumbnail)

class TargetCanvas(QWidget):
    def __init__(self, target):
        super(TargetCanvas, self).__init__(None)
        self.target = target

        self.setMaximumHeight(70)
        self.setMinimumHeight(70)
        self.setMinimumWidth(70)

        self.setImage(target.rois[0].thumbnail)

    def setImage(self, image):
        new_height = self.geometry().height()-10
        new_width = int(new_height*image.shape[1]/image.shape[0])
        self.qImage = cvImgToQImg(cv2.resize(image, (new_width, new_height)))

    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)

        if self.target.get_confidence() >= 3:
            painter.setBrush(QColor(0, 255, 0))
            painter.drawRect(QRect(-1, -1, self.geometry().width()+2, self.geometry().height()+2))

        painter.drawImage(5, 5, self.qImage)
        painter.drawText(200, 20, "{} | {}".format(self.target.get_shape(), self.target.get_alphanumeric()))
        painter.drawText(200, 40, "{} | {}".format(self.target.coord[0], self.target.coord[1]))

def cvImgToQImg(cvImg):
    return QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], cvImg.strides[0], QImage.Format_RGB888)
