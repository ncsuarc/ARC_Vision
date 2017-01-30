import ARC
import cv2
import numpy as np
import argparse
import filters

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Display', 1920, 1080)

parser = argparse.ArgumentParser(description='Search flight images for targets.')
parser.add_argument("-i", "--input", required=True, help="Flight number to search")
args = vars(parser.parse_args())

flight = ARC.Flight(args['input'])
targets = flight.all_targets()
images = []

for tgt in targets:
    if not ((tgt.target_type == 0) or (tgt.target_type == 1) or (tgt.target_type == None)):
        continue
    new_images = flight.images_near(tgt.coord, 50)
    images.extend(new_images)
#Remove duplicate files
images = dict((image.filename, image) for image in images).values()
#images = flight.all_images()

n = 0

for image_file in images:
    filename = image_file.filename
    image = cv2.imread(filename[:-3] + 'jpg')
    ROIs = filters.high_pass_filter(image_file)
    ROIs = filters.false_positive_filter(ROIs)
    for roi in ROIs:
        cv2.imshow('Display', roi.roi)
        cv2.waitKey()
        #cv2.imwrite("roi/roi{}.jpg".format(n), roi.roi)
        n += 1
