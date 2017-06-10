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
        cv2.imwrite(location, cv2.cvtColor(self.roi.roi, cv2.COLOR_BGR2RGB))

    def saveThumbnailImage(self, location):
        cv2.imwrite(location, cv2.cvtColor(self.roi.thumbnail, cv2.COLOR_BGR2RGB))

class TargetCanvas(QWidget):
    def __init__(self, target):
        super(TargetCanvas, self).__init__(None)
        self.target = target

        self.setMaximumHeight(70)
        self.setMinimumHeight(70)
        self.setMinimumWidth(70)

        self.n = 0
        self.setImage(target.rois[self.n])

        self.target.remove_target.connect(self.scheduleDeleteWidget)

    def scheduleDeleteWidget(self):
        self.setParent(None)
        self.deleteLater()

    def setImage(self, roi):
        image = roi.thumbnail
        new_height = self.geometry().height()-10
        new_width = int(new_height*image.shape[1]/image.shape[0])
        self.qImage = cvImgToQImg(cv2.resize(image, (new_width, new_height)))
        self.qImageShape = cvImgToQImg(cv2.resize(roi.shape_img, (new_width, new_height)))
        self.qImageChar = cvImgToQImg(cv2.resize(roi.alphanumeric_img, (new_width, new_height)))

    def mousePressEvent(self, evt):
        super(TargetCanvas, self).mousePressEvent(evt)
        self.n += 1
        if self.n == len(self.target.rois):
            self.n = 0
        self.setImage(self.target.rois[self.n])
        self.repaint()

    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)

        painter.drawImage(5, 5, self.qImage)
        painter.drawImage(150, 5, self.qImageShape)
        painter.drawImage(300, 5, self.qImageChar)
        painter.drawText(400, 20, "{} | {}".format(self.target.get_shape(), self.target.get_alphanumeric()))
        painter.drawText(400, 40, "{} | {}".format(self.target.coord[0], self.target.coord[1]))
        painter.drawText(400, 60, "N: {}".format(self.target.get_confidence()))

def cvImgToQImg(cvImg):
    return QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], cvImg.strides[0], QImage.Format_RGB888)
