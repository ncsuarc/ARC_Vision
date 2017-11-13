from .nn import Model

import numpy as np
import cv2

print("Loading False Positive model...")
fp_model = Model('models/false_positive')
print("Loading Shape model...")
shape_model = Model('models/shape')
print("Loading Alphanumeric model...")
alphanumeric_model = Model('models/alphanumeric')

def check_targets(images):
    labels = fp_model.test([cv2.resize(image, (64, 64)).flatten() for image in images])
    return [bool(np.argmax(label)) for label in labels]

def classify_shape(image):
    return shape_model.classify([cv2.resize(image, (64, 64)).flatten()])[0]

def classify_alphanumeric(image):
    return alphanumeric_model.classify([cv2.resize(image, (64, 64)).flatten()])[0]

def get_target_info(img):
    #Use KMeans to segment the image into distinct colors
    #K = 3 for background, target, and letter
    K = 3
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS, 0, 0.25)
    compactness, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS )
    centers = np.uint8(centers)

    #Determine which class is the background
    idx = np.argmin(np.mean(centers, axis=1))
    centers[idx] = [0, 0, 0]
    labels = labels.flatten()

    #Determine which classes are not the background
    indexes = [i for i in range(K)]
    indexes.remove(idx)
    first_color_index = indexes[0]
    second_color_index = indexes[1]

    #Create two separate images, one for each non-background class
    first_labels = np.copy(labels)
    first_labels[first_labels == second_color_index] = idx
    first_img = centers[first_labels]
    first_img = first_img.reshape((img.shape))

    second_labels = np.copy(labels)
    second_labels[second_labels == first_color_index] = idx
    second_img = centers[second_labels]
    second_img = second_img.reshape((img.shape))

    #Determine shape vs letter
    (_, first_contours, _) = cv2.findContours(cv2.cvtColor(first_img, cv2.COLOR_RGB2GRAY),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (_, second_contours, _) = cv2.findContours(cv2.cvtColor(second_img, cv2.COLOR_RGB2GRAY),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if(cv2.contourArea(first_contours[0]) > cv2.contourArea(second_contours[0])):
        shape_cnt = first_contours[0]
        shape_color = centers[first_color_index]
        shape_image = first_img

        letter_color = centers[second_color_index]
        letter_img = second_img
    else:
        letter_color = centers[first_color_index]
        letter_img = first_img

        shape_cnt = second_contours[0]
        shape_color = centers[second_color_index]
        shape_image = second_img

    letter_mask = cv2.inRange(letter_img, letter_color, letter_color)

    shape_mask = np.zeros(img.shape[0:2], np.uint8)
    cv2.drawContours(shape_mask, [shape_cnt], 0, 255, -1)
    return ((shape_mask, shape_color), (letter_mask, letter_color))

def draw_mask_color(mask, color):
    img = np.zeros(mask.shape + (len(color),), np.uint8)

    img[:,:] = color

    return cv2.bitwise_and(img, img, mask=mask)
