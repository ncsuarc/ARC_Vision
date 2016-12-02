import ARC
import cv2
import numpy as np
from filters import *
import time

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Display', 1920, 1080)

#flight = ARC.Flight(95) #2015 Competition
flight = ARC.Flight(157) #2016 Competition

targets = flight.all_targets()
images = []

for tgt in targets:
    if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
        continue
    dist = 10
    while True:
        new_images = flight.images_near(tgt.coord, dist)
        if new_images:
            break
        dist += 1
    images.extend(new_images)
images = flight.all_images()
n=0
for image_file in images:
#for i in range(1):
#    image_file = images[5]
    if(image_file.nadired):
        ROIs = filter_primary(image_file)

        for region in ROIs:
            if(check_target(region.roi)):
                cv2.imshow('Display', region.roi)
                if(cv2.waitKey() == 115):
                    #cv2.imwrite("crops/image"+str(n)+".jpg", image)
                    n += 1
