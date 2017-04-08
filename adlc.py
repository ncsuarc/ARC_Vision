import ARC

import filters
import roi

import atexit
import os

from collections import deque

from PyQt5.QtCore import (QObject, QRunnable, QThreadPool, QTimer, pyqtSignal)

class ADLCProcessor(QObject):

    new_roi = pyqtSignal('PyQt_PyObject')
    new_target = pyqtSignal('PyQt_PyObject')

    def __init__(self, flight_number=0, threads=4):
        super(ADLCProcessor, self).__init__()

        self.flight_number = flight_number
        self.threads = threads

        self.flight = ARC.Flight(flight_number)

        self.lq_listener = ARC.db.Listener(self.flight.database, ARC.dbschema.notify_image_low_quality)
        self.hq_listener = ARC.db.Listener(self.flight.database, ARC.dbschema.notify_image_high_quality)

        self.queryNewImagesTimer = QTimer()
        self.queryNewImagesTimer.timeout.connect(self.queryNewImages)
        self.queryNewImagesTimer.start(500)

        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(threads)

        self.images = deque(self.flight.all_images()) 

        self.queueCount = 0
        
        self.rois = []
        self.targets = []

        atexit.register(self.cleanup)

    def cleanup(self):
        print('ADLC Processor cleaning up...')
        self.pool.waitForDone()

    def getQueueLength():
        return self.queueCount + len(self.images)

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
            #while not self.images[0].nadired:
            #    self.images.popleft() #Throw out images that are not nadired
            self.startImageProcessing(self.images.popleft())

    def startImageProcessing(self, image):
        processor = ImageProcessor(image, self.processingFinished, self.newTarget)
        self.pool.start(processor)
        self.queueCount += 1

    def processingFinished(self):
        self.queueCount -= 1

    def newTarget(self, new_roi):
        self.rois.append(new_roi)
        self.new_roi.emit(new_roi)
        for t in self.targets:
            if t.is_duplicate(new_roi):
                return
        tgt = roi.Target(new_roi)
        self.targets.append(tgt)
        self.targets = sorted(self.targets, key=lambda x: x.get_confidence(), reverse=True) 
        self.new_target.emit(tgt)

class ImageProcessorConnector(QObject):

    finished = pyqtSignal()
    new_target = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        super(ImageProcessorConnector, self).__init__()

class ImageProcessor(QRunnable): 
    def __init__(self, image, finished_callback, new_target_callback):
        super(ImageProcessor, self).__init__()
        
        self.setAutoDelete(True)

        self.image = image
        self._emitter = ImageProcessorConnector()
        self._emitter.finished.connect(finished_callback)
        self._emitter.new_target.connect(new_target_callback)

    def run(self):
        try:
            rois = filters.get_targets(self.image)
            for roi in rois:
                self._emitter.new_target.emit(roi)
            self._emitter.finished.emit()
        except Exception as e:
            print(e)

if __name__=="__main__":
    import sys
    import argparse
    from PyQt5.QtCore import QCoreApplication
    import signal

    def sigint_handler(*args):
        QCoreApplication.quit()

    signal.signal(signal.SIGINT, sigint_handler)

    parser = argparse.ArgumentParser(description='Search flight images for targets.')
    parser.add_argument("-i", "--input_flight", help="Flight number to search")
    args = parser.parse_args()
    
    app = QCoreApplication(sys.argv)
    processor = ADLCProcessor(flight_number=args.input_flight)

    global target_count
    target_count = 0

    global roi_count
    roi_count = 0

    def new_target(roi):
        global target_count
        target_count += 1

    def new_roi(roi):
        global roi_count
        roi_count += 1

    processor.new_target.connect(new_target)
    processor.new_roi.connect(new_roi)

    app.exec_()

    print("Found %d targets in %d rois." % (target_count, roi_count))
