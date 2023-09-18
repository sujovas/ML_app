import cv2
import math

import numpy as np
from patchify import patchify


def defBoundingBox(image):
    image = cv2.bitwise_not(image)
    x, y, w, h = cv2.boundingRect(image)

    for k in range(20):
        if k * 256 > w:
            break
    for l in range(20):
        if l * 256 > h:
            break

    x_start = (x - math.ceil((k * 256 - w) / 2))
    y_start = (y - math.floor((l * 256 - h) / 2))
    x_end = (x + w + math.floor((k * 256 - w) / 2))
    y_end = (y + h + math.ceil((l * 256 - h) / 2))
    if y_start < 0:
        y_end = y_end - y_start
        y_start = 0

    if x_start < 0:
        x_end = x_end - x_start
        x_start = 0

    h = y_end - y_start
    w = x_end - x_start

    return x_start, y_start, x_end, y_end


def preprocessImage(inputSize, large_img):


    gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    box_coordinates = defBoundingBox(thresh)

    crop_img = gray[box_coordinates[1]:box_coordinates[3], box_coordinates[0]:box_coordinates[2]]

    large_image = crop_img / 255.
    patches = patchify(large_image, (256, 256), step=256)
    predict_patch = np.ndarray([patches.shape[0], patches.shape[1], inputSize[0], inputSize[0]])

    return large_image.shape, patches, predict_patch, crop_img



