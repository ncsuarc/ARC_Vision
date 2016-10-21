import ARC
import cv2
import numpy as np
from filters import *
import time

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Display', 1920, 1080)

flight = ARC.Flight(157)

#targets = flight.all_targets()
#images = []
#
#for tgt in targets:
#    if not ((tgt.target_type == 0) or (tgt.target_type == 1)):
#        continue
#    dist = 10
#    while True:
#        new_images = flight.images_near(tgt.coord, dist)
#        if new_images:
#            break
#        dist += 1
#    images.extend(new_images)

images = flight.all_images()


for image_file in images:
#for i in range(1):
#    image_file = images[0]
    filename = image_file.filename
    image = cv2.imread(filename[:-3] + 'jpg')
    imgs = filter_primary(image)


    for img in imgs:
        if(check_target(img)):
            cv2.imshow('Display', image)
            cv2.waitKey()
            cv2.imshow('Display', img)
            cv2.waitKey()
