import cv2
import os

import matplotlib.pyplot as plt

from PIL import Image
from utils import inputSize, loadModel
from preprocess import preprocessImage
from detection import detectShots, reconstruct
from postprocess import findCentroid, concatenateMasks, findContourShots, findContoursCircles, calculateShortestDistance, definePoints

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# flask app
def process_file(image):
    # Get the file name and extension
    photo = image
    # photo = cv2.imread("..\\IMG_4348.png", 1)

    size = inputSize()
    model = loadModel(size)

    shape, patches, predict_patch, original_cropped = preprocessImage(inputSize=size, large_img=photo)
    predicted_patches = detectShots(patches, predict_patch, model)
    reconstructed_mask = reconstruct(predicted_patches, original_cropped.shape)
    thresh, segmentation, centroid, contour = findCentroid(original_cropped, reconstructed_mask)
    mask, contours = concatenateMasks(thresh, segmentation, centroid)
    contours_shots = findContourShots(mask, contour)
    circle_radius, mask = findContoursCircles(contour, centroid, mask)
    shortest_shot_distances = calculateShortestDistance(centroid, contours_shots)
    points = definePoints(circle_radius, shortest_shot_distances)

    return mask, points



# photo = cv2.imread("/Users/sarasujova/PycharmProjects/Priloha_A/IMG_4348.png", 1)
#
# size = inputSize()
# model = loadModel(size)
#
# shape, patches, predict_patch, original_cropped = preprocessImage(inputSize=size, large_img=photo)
# predicted_patches = detectShots(patches, predict_patch, model)
# reconstructed_mask = reconstruct(predicted_patches, original_cropped.shape)
# thresh, segmentation, centroid, contour = findCentroid(original_cropped, reconstructed_mask)
# mask, contours = concatenateMasks(thresh, segmentation, centroid)
# contours_shots = findContourShots(mask, contour)
# circle_radius, mask = findContoursCircles(contour, centroid, mask)
# shortest_shot_distances = calculateShortestDistance(centroid, contours_shots)
# points = definePoints(circle_radius, shortest_shot_distances)
#
# plt.figure(figsize=(2, 2))
# plt.subplot(222)
# plt.title('Prediction of large Image')
# plt.imshow(reconstructed_mask, cmap='gray')
# plt.subplot(223)
# plt.title('Mask apply')
# plt.imshow(mask, cmap='gray')
# plt.show()