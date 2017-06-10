import ARC

import filters
import roi
import classify

import atexit
import os

from collections import deque
from osgeo import ogr
from interop import InterOp

from PyQt5.QtCore import (QObject, QRunnable, QThreadPool, QTimer, QSettings, pyqtSignal)

import time

current_milli_time = lambda: int(round(time.time() * 1000))

class ADLCProcessor(QObject):

    new_roi = pyqtSignal('PyQt_PyObject')
    new_target = pyqtSignal('PyQt_PyObject')
    processing_finished = pyqtSignal()

    TARGET_THRESHOLD = 3
    TARGET_OVERLOAD = 4

    def __init__(self, flight_number=0, threads=4, check_interop=True):
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
        self.potential_targets = []

        atexit.register(self.cleanup)
        
        self.check_interop = check_interop

        if self.check_interop:
            settings = QSettings("ARC", "PCC Interop Plugin")
            ip = settings.value('host')
            port = settings.value('port')
            username = settings.value('username')
            password = settings.value('password')
            io = InterOp(username, password, ip, port)
            missions = io.get_missions()
            for mission in missions:
                if bool(mission.get('active')):
                    active_mission = mission
                    break
            interop_grid_points = active_mission.get('search_grid_points')

            grid_points = [None] * len(interop_grid_points)

            for point in interop_grid_points: 
                grid_points[point.get('order') - 1] = (point.get('latitude'), point.get('longitude'))

            ring = ogr.Geometry(ogr.wkbLinearRing)

            for point in grid_points:
                ring.AddPoint(point[0], point[1])

            ring.AddPoint(grid_points[0][0], grid_points[0][1])
            self.search_grid = ogr.Geometry(ogr.wkbPolygon)  
            self.search_grid.AddGeometry(ring)

        self.last_image_time = current_milli_time()

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
            if self.check_interop:
                while len(self.images) > 0:
                    image = self.images.popleft()
                    for rel_coord in [(0,0), (0, image.height), (image.width, 0), (image.width, image.height)]:
                        point = ogr.Geometry(ogr.wkbPoint)
                        coord = image.coord(*rel_coord)
                        point.AddPoint(*coord)
                        if(self.search_grid.Contains(point)):
                            self.startImageProcessing(image)
                            self.checkProcessingFinished()
                            return
            else:
                self.startImageProcessing(self.images.popleft())
        self.checkProcessingFinished()
    
    def checkProcessingFinished(self):
        if len(self.images) == 0:
            print('Queue empty')
            delta_t = current_milli_time() - self.last_image_time
            if delta_t > 60000: #One minute without new images
                self.processing_finished.emit()
            return
        else:
            self.last_image_time = current_milli_time()

    def startImageProcessing(self, image):
        processor = ImageProcessor(image, self.processingFinished, self.newTarget)
        self.pool.start(processor)
        self.queueCount += 1

    def processingFinished(self):
        self.queueCount -= 1

    def newTarget(self, new_roi):
        self.rois.append(new_roi)
        self.new_roi.emit(new_roi)
        for t in self.potential_targets:
            if t.is_duplicate(new_roi):
                if t in self.targets:
                    if(t.get_confidence() > ADLCProcessor.TARGET_OVERLOAD):
                        #self.targets.remove(t)
                        print('REMOVING TARGET %d' % t.get_confidence())
                        #t.remove_target.emit()
                else:
                    if(t.get_confidence() >= ADLCProcessor.TARGET_THRESHOLD):
                        self.targets.append(t)
                        self.new_target.emit(t)
                return
        tgt = roi.Target(new_roi)

        self.potential_targets.append(tgt)
        self.potential_targets = sorted(self.potential_targets, key=lambda x: x.get_confidence(), reverse=True) 

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
    parser.add_argument("-i", "--input-flight", help="Flight number to search")
    parser.add_argument("--no-interop", action="store_true")
    args = parser.parse_args()
    
    app = QCoreApplication(sys.argv)
    processor = ADLCProcessor(flight_number=args.input_flight, check_interop=(not args.no_interop))

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

    def finished():
        QCoreApplication.quit()

    processor.new_target.connect(new_target)
    processor.new_roi.connect(new_roi)
    processor.processing_finished.connect(finished)

    app.exec_()

    print("Found %d targets in %d rois." % (target_count, roi_count))
