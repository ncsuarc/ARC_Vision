import ARC
import cv2
import numpy as np
from filters import *
import time

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.namedWindow('Unrotated', cv2.WINDOW_NORMAL)
cv2.namedWindow('Rotated', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Display', 1920, 1080)
cv2.resizeWindow('Unrotated', 800, 500)
cv2.resizeWindow('Rotated', 800, 450)
#flight = ARC.Flight(95) #2015 Competition
#flight = ARC.Flight(157) #2016 Competition
flight = ARC.Flight(162)

targets = flight.all_targets()
images = []

for tgt in targets:
    if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
        continue
    dist = 50
#    while True:
    new_images = flight.images_near(tgt.coord, dist)
#        if new_images:
#            break
#        dist += 1
    images.extend(new_images)
#images = flight.all_images()

n=0

for image_file in images:
    filename = image_file.filename
    image = cv2.imread(filename[:-3] + 'jpg')
#    cv2.imshow('Display', image)
#    cv2.waitKey()
    ROIs = filter_primary(image_file)
#    for region in ROIs:
#        cv2.imshow('Display', region.roi)
#        cv2.waitKey()
