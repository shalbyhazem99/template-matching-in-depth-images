import cv2 as cv
from cv2 import COLOR_RGB2GRAY
import numpy as np
import numpy.ma as ma
import math
from customRansac import doIntersect


def applyCannyFilter(img_scene, depth_image, sift_mask, keypoints_scene, descriptors_scene):
    # Apply Canny filter on depth image
    depth_denoised = cv.fastNlMeansDenoising(depth_image, None, 80, 7, 21)
    u8_denoised = (depth_denoised*255).astype(np.uint8)
    u8_denoised = cv.bitwise_and(u8_denoised, u8_denoised, mask=sift_mask)
    edges = cv.Canny(u8_denoised, 100, 170, None, 3)

    # Apply Canny filter on RGB image
    gray_scene = cv.cvtColor(img_scene, COLOR_RGB2GRAY)
    blurred_scene = cv.medianBlur(gray_scene, 9)
    scene_edges = cv.Canny(blurred_scene, 30, 150, None, 3)

    depth_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    scene = cv.cvtColor(scene_edges, cv.COLOR_GRAY2BGR)
    scene_copy = np.copy(scene)
    scene_copy2 = np.copy(scene)

    # Copia per mostrare tutti i keypoints
    cdst_copy = np.copy(depth_edges)
    # for kp in keypoints_scene:
    #	cv.circle(cdst_copy, (int(kp.pt[0]), int(kp.pt[1])), radius=0, color=(255, 0, 0), thickness=-1)

    # Find the lines on the images
    lines_depth = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 70, 40)
    lines_scene = cv.HoughLinesP(scene_edges, 1, np.pi/180, 50, None, 30, 30)

    # Show Hough lines on the image
    image_to_show = drawHoughLines(lines_depth, depth_edges)
    cv.imshow('Lines', image_to_show)

    image_to_show = drawHoughLines(lines_scene, scene)
    cv.imshow('Lines_scene', image_to_show)

    final_lines = []
    uncertainty = 15

    final_lines, lines_scene = selectEdgesFoundInDepthAndRGBImages(
        lines_scene, lines_depth, uncertainty)

    length = len(final_lines)
    final_lines, lines_scene = addPerpendicularLines(
        lines_scene, final_lines, uncertainty)

    # Check if the previous added lines have lines perpendicular to themself
    if length < len(final_lines):
        final_lines, lines_scene = addOtherPerpendicularLines(
            length, final_lines, lines_scene, uncertainty)

    image_to_show = drawHoughLines(final_lines, scene_copy, 1)
    cv.imshow('Final lines', image_to_show)

    dist_threshold = 50
    final_kp_scene, final_des_scene = findKeypointsWithSomeDistance(
        keypoints_scene, final_lines, dist_threshold, depth_edges, descriptors_scene)
    # final_kp_scene, final_des_scene = findKeypointsInTheRectangles(
    #     keypoints_scene, final_lines, depth_edges, descriptors_scene, uncertainty)

    descriptors_scene = np.asarray(final_des_scene)
    keypoints_scene = np.asarray(final_kp_scene)

    for kp in keypoints_scene:
        cv.circle(cdst_copy, (int(kp.pt[0]), int(
            kp.pt[1])), radius=1, color=(0, 255, 0), thickness=-1)
    cv.imshow('Keypoints found', cdst_copy)

    # Print number of keypoints after applying the filter
    print("AFTER FILTER: " + str(len(final_kp_scene)))
    return descriptors_scene, keypoints_scene,


# Draw Hough lines on an image
def drawHoughLines(lines, image, n_param=2):
    if lines is not None:
        for i in range(0, len(lines)):
            if n_param == 1:
                l = lines[i]
            else:
                l = lines[i][0]
            cv.line(image, (l[0], l[1]), (l[2], l[3]),
                    (0, 0, 255), 1, cv.LINE_AA)
    return image


# Extract extremities and angle of the given line
def extractInfoFromTheLine(line):
    p1 = (line[0], line[1])
    p1 = np.asarray(p1)

    p2 = (line[2], line[3])
    p2 = np.asarray(p2)

    # slope of the line
    theta = math.atan2((p2[1] - p1[1]),
                       (p2[0] - p1[0]))  # atan2 dà l'angolo in rad
    return p1, p2, theta


# Select edges that were found both on depth image and RGB image
def selectEdgesFoundInDepthAndRGBImages(lines_scene, lines_depth, uncertainty):
    i = 0
    final_lines = []
    while i < len(lines_scene):

        l = lines_scene[i][0]
        l_p1, l_p2, theta = extractInfoFromTheLine(l)
        equal = False

        for j in range(0, len(lines_depth)):

            m = lines_depth[j][0]
            m_p1, m_p2, m_theta = extractInfoFromTheLine(m)

            # Check if the lines are parallel
            # Which means roughly "if angle and desired_angle are within 6 radiants of each other, disregarding direction"
            if (math.fabs(math.cos(theta - m_theta)) > 0.99452189536):  # math.cos() vuole l'angolo in rad
                # Check if the two lines intersect with each other, otherwise check if an extremity is closed to another with a certain uncertainty
                if doIntersect(l_p1, l_p2, m_p1, m_p2) or (
                    math.fabs(l_p1[0] - m_p1[0]) < uncertainty and math.fabs(l_p1[1] - m_p1[1]) < uncertainty) or (
                        math.fabs(l_p2[0] - m_p2[0]) < uncertainty and math.fabs(l_p2[1] - m_p2[1]) < uncertainty):
                    equal = True
                    break

        if equal:
            final_lines.append(l)
            lines_scene = np.delete(lines_scene, i, axis=0)
            i -= 1
        i += 1
    return final_lines, lines_scene


# Add edges that are perpendicular to the lines already selected
def addPerpendicularLines(lines_scene, final_lines, uncertainty):
    i = 0
    for i in range(0, len(final_lines)):

        l = final_lines[i]
        l_p1, l_p2, theta = extractInfoFromTheLine(l)
        equal = 0

        for j in range(0, len(lines_scene)):

            m = lines_scene[j][0]
            m_p1, m_p2, m_theta = extractInfoFromTheLine(m)

            # Check if the two lines are perpendicular
            # Which means roughly "if angle and desired_angle are within 6 radiants of each other, disregarding direction"
            # 0.121869343405 (cos(83 rad))
            if (math.fabs(math.cos(theta - m_theta)) < 0.3421654):
                # Check if two estremities are closed with a certain uncertainty
                if (math.fabs(l_p1[0] - m_p1[0]) < uncertainty and math.fabs(l_p1[1] - m_p1[1]) < uncertainty) or (
                        math.fabs(l_p2[0] - m_p2[0]) < uncertainty and math.fabs(l_p2[1] - m_p2[1]) < uncertainty) or (
                            math.fabs(l_p1[0] - m_p2[0]) < uncertainty and math.fabs(l_p1[1] - m_p2[1]) < uncertainty) or (
                                math.fabs(l_p2[0] - m_p1[0]) < uncertainty and math.fabs(l_p2[1] - m_p1[1]) < uncertainty):
                    equal += 1
                    if equal == 1:
                        m1 = m
                        j1 = j
                    elif equal == 2:
                        m2 = m
                        j2 = j
                        break

        if equal > 0:
            final_lines.append(m1)
            lines_scene = np.delete(lines_scene, j1, axis=0)
            if equal == 2:
                final_lines.append(m2)
                lines_scene = np.delete(lines_scene, j2 - 1, axis=0)

    return final_lines, lines_scene


# Add edges that are perpendicular to previous added lines
def addOtherPerpendicularLines(length, final_lines, lines_scene, uncertainty):
    for i in range(length, len(final_lines)):

        l = final_lines[i]
        l_p1, l_p2, theta = extractInfoFromTheLine(l)
        equal = False

        for j in range(0, len(lines_scene)):

            m = lines_scene[j][0]
            m_p1, m_p2, m_theta = extractInfoFromTheLine(m)

            # Check if the two lines are perpendicular
            # Which means roughly "if angle and desired_angle are within 6 radiants of each other, disregarding direction"
            if (math.fabs(math.cos(theta - m_theta)) < 0.322):  # 0.121869343405
                # Check if two estremities are closed with a certain uncertainty
                if (math.fabs(l_p1[0] - m_p1[0]) < uncertainty and math.fabs(l_p1[1] - m_p1[1]) < uncertainty) or (
                        math.fabs(l_p2[0] - m_p2[0]) < uncertainty and math.fabs(l_p2[1] - m_p2[1]) < uncertainty) or (
                            math.fabs(l_p1[0] - m_p2[0]) < uncertainty and math.fabs(l_p1[1] - m_p2[1]) < uncertainty) or (
                                math.fabs(l_p2[0] - m_p1[0]) < uncertainty and math.fabs(l_p2[1] - m_p1[1]) < uncertainty):
                    equal = True
                    break

        if equal:
            final_lines.append(m)
            lines_scene = np.delete(lines_scene, j, axis=0)
    return final_lines, lines_scene


# Find all the keypoints that are close to the selected lines
def findKeypointsWithSomeDistance(keypoints_scene, final_lines, dist_threshold, depth_edges, descriptors_scene):
    final_kp_scene = []
    final_des_scene = []

    for j in range(0, len(keypoints_scene)):
        kp = keypoints_scene[j]

        p3 = (int(kp.pt[0]), int(kp.pt[1]))
        p3 = np.asarray(p3)

        near = False
        for i in range(0, len(final_lines)):
            l = final_lines[i]

            # P1 --> (l[0], l[1]) P2 --> (l[2], l[3]), P3 --> (kp.pt[0],kp.pt[1])
            p1, p2, theta = extractInfoFromTheLine(l)

            # Calculate the distance between P1-P2 segment and P3 point
            diff_x = p2[0] - p1[0]
            diff_y = p2[1] - p1[1]

            norm = diff_x**2 + diff_y**2

            u = ((p3[0] - p1[0]) * diff_x +
                 (p3[1] - p1[1]) * diff_y) / float(norm)

            if u > 1:
                u = 1
            elif u < 0:
                u = 0

            x = p1[0] + u * diff_x
            y = p1[1] + u * diff_y

            dx = x - p3[0]
            dy = y - p3[1]

            distance = (dx*dx + dy*dy)**.5

            # Check if the distance is equal or less than the threshold
            if(distance <= dist_threshold):
                near = True
                #print("Punto P1: " + str(p1) + " || Punto P2: " + str(p2) + " || Punto P3: " + str(p3) + " || Distance: " + str(distance))
                break

        if near:
            final_kp_scene.append(kp)
            final_des_scene.append(descriptors_scene[j])
    return final_kp_scene, final_des_scene


# Find all the keypoints that are inside the found rectangles
# def findKeypointsInTheRectangles(keypoints_scene, final_lines, depth_edges, descriptors_scene, uncertainty):
#     final_kp_scene = []
#     final_des_scene = []

#     rectangles = createRectangles(final_lines, uncertainty)

#     return final_kp_scene, final_des_scene


# Given the final lines found, it creates an array of rectangles
# def createRectangles(final_lines, uncertainty):
    # rectangles = []
    # tempRect = []
    # tempRect.append(final_lines[0])
    # p1, p2, theta = extractInfoFromTheLine(final_lines[0])
    # final_lines.remove(final_lines[0])
    # for line in final_lines:
    #     try:
    #         tempRect.index(line)
    #     except ValueError:
    #         l_p1, l_p2, l_theta = extractInfoFromTheLine(line)
    #         if (math.fabs(p1[0] - l_p1[0]) < uncertainty and math.fabs(p1[1] - l_p1[1]) < uncertainty) or (
    #             math.fabs(p2[0] - l_p2[0]) < uncertainty and math.fabs(p2[1] - l_p2[1]) < uncertainty) or (
    #                 math.fabs(p1[0] - l_p2[0]) < uncertainty and math.fabs(p1[1] - l_p2[1]) < uncertainty) or (
    #                     math.fabs(p2[0] - l_p1[0]) < uncertainty and math.fabs(p2[1] - l_p1[1]) < uncertainty):
    #             tempRect.append(line)
    #             final_lines.remove(line)

    # return rectangles
