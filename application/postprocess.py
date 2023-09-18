import cv2

import numpy as np

from statistics import mean
from math import ceil


def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    if len(p2) == 1:
        p2 = tuple(p2[0])
    if len(p1) == 1:
        p1 = tuple(p1[0])
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


def findCentroid(large_img, segmentation):
    segmentation=cv2.bitwise_not(segmentation)
    blur = cv2.GaussianBlur(large_img, (71, 71), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((39, 39), np.uint8)
    area_closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.floodFill(area_closing, None, seedPoint=(0, 0), newVal=0, loDiff=1, upDiff=1)

    contour, hierarchy = cv2.findContours(area_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find centroid
    for c in contour:
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print('centroid: X:{}, Y:{}'.format(cX, cY))
        centroid = (cX, cY)

    return thresh, segmentation, centroid, contour


def concatenateMasks(thresh, segmentation, centroid):
    mask = np.full(thresh.shape, 0, "uint8")
    segmentation = cv2.threshold(segmentation, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel_borderline = np.ones((39, 39), np.uint8)
    area_closing_borderline = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_borderline)
    contours, hierarchies = cv2.findContours(area_closing_borderline, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours[1], -1, 255, 5)
    cv2.floodFill(mask, None, seedPoint=(0, 0), newVal=255, loDiff=1, upDiff=1)

    mask = cv2.bitwise_not(mask)

    mask2 = cv2.bitwise_and(mask, segmentation)
    mask3 = cv2.bitwise_and(mask, cv2.bitwise_not(mask2))

    return mask3, contours


def findContourShots(mask, contour):
    contours_shots = []
    mask = np.uint8(mask)
    contours_points, hierarchies_points = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for point in contours_points:
        if point.size < 500 and point.size > 5:
            contours_shots.append(point)

    cv2.drawContours(mask, contour[0], -1, 0, 5)

    return contours_shots, mask


def findContoursCircles(contour, centroid, mask):

    cv2.circle(mask, centroid, 5, (0, 0, 0), -1)

    distances = []
    contour = contour[0]
    for i in range(len(contour)):
        distances.append(distanceCalculate(centroid, tuple(map(tuple, contour[i]))[0]))
    distance_avg = mean(distances)
    one_distance = ceil(distance_avg / 4)
    circle_radius = [one_distance, one_distance * 2, one_distance * 3, one_distance * 4, one_distance * 5,
                     one_distance * 6, one_distance * 7,
                     one_distance * 8, one_distance * 9, one_distance * 10]
    for j in range(len(circle_radius)):
        cv2.circle(mask, centroid, radius=circle_radius[j], color=0, thickness=1)

    return circle_radius, mask


def calculateShortestDistance(centroid, contours_shots):
    shortest_shot_distances = []
    for shot in contours_shots[0]:
        shot_distances = []
        for shot_point in shot:
            shot_distances.append(distanceCalculate(centroid, shot_point))
        if shot_distances:
            shortest_shot_distance = np.amin(shot_distances)
            shortest_shot_distances.append(shortest_shot_distance)
        else:
            shortest_shot_distances.append(0)
    return shortest_shot_distances

def definePoints(circle_radius, shortest_shot_distances):
##evaluate distances
    points_dict = [
        {"points": 10, "distance_short": 0, "distance_long": circle_radius[0]},
        {"points": 9, "distance_short":circle_radius[0], "distance_long": circle_radius[1]},
        {"points": 8, "distance_short":circle_radius[1], "distance_long": circle_radius[2]},
        {"points": 7, "distance_short":circle_radius[2], "distance_long": circle_radius[3]},
        {"points": 6, "distance_short":circle_radius[3], "distance_long": circle_radius[4]},
        {"points": 5, "distance_short":circle_radius[4], "distance_long": circle_radius[5]},
        {"points": 4, "distance_short":circle_radius[5], "distance_long": circle_radius[6]},
        {"points": 3, "distance_short":circle_radius[6], "distance_long": circle_radius[7]},
        {"points": 2, "distance_short":circle_radius[7], "distance_long": circle_radius[8]},
        {"points": 1, "distance_short":circle_radius[8], "distance_long": circle_radius[9]},
    ]

    shot_points = []
    for distance in shortest_shot_distances:
        found = False
        for item in points_dict:
            if item["distance_short"] <= distance < item["distance_long"]:
                shot_points.append(item["points"])
                found = True
                break
        if not found:
            shot_points.append(0)
    return shot_points


# result = cv2.bitwise_or(thresh, thresh, segmentation)


