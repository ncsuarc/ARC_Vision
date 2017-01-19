import ARC
import cv2
import numpy as np
from filters import *
import time

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Display', 1920, 1080)
#flight = ARC.Flight(95) #2015 Competition
#flight = ARC.Flight(157) #2016 Competition
flight = ARC.Flight(162) #1-15-17 Flight Test

targets = flight.all_targets()
images = []

for tgt in targets:
    if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
        continue
    new_images = flight.images_near(tgt.coord, 50)
    images.extend(new_images)
#images = flight.all_images()

n = 0

for image_file in images:
    filename = image_file.filename
    image = cv2.imread(filename[:-3] + 'jpg')
    ROIs = high_pass_filter(image_file)
