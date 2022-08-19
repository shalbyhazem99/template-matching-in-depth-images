from ast import While
import cv2 as cv
from cv2 import COLOR_RGB2GRAY
import numpy as np
import numpy.ma as ma
import math
from canny_filter_method import applyCannyFilter
from customRansac import customFindHomography, buildKDTree
from customRansac import customFindHomographyPlane3D
from customRansac import customFindHomographyNormalSampling3D
from customRansac import customFindHomography3DTree
from customRansac import customFindHomographyDimension
import time
import json


img_scene = cv.imread("3D/5/rgb_image.jpg")
point_cloud = np.load("3D/5/pointCloud.npy")
depth_image = cv.imread("3D/5/depth_image.jpg", cv.IMREAD_GRAYSCALE)

template = "barchette.png"
template_path = "Templates/" + template
img_object = cv.imread(template_path)
cv.imshow('Point cloud', point_cloud)

with open('dimensions.json', 'r') as f:
    data = json.load(f)

json_template = template.rpartition('.')[0]
# Template dimension [larghezza, altezza, profondità] --> [x, y, z]
dimension = [data[json_template]["width"], data[json_template]
             ["height"], data[json_template]["depth"]]

# Detect the keypoints using SIFT Detector, compute the descriptors
minHessian = 1
detector = cv.SIFT_create()

# Creation of the mask to delete the image edges so SIFT doesn't look there, too noisy
##
# https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift
##
left_border = 50
right_border = img_scene.shape[1] - 50
top_border = 20
bottom_border = img_scene.shape[0] - 20
sift_mask = np.zeros(img_scene.shape[:2], np.uint8)
cv.rectangle(sift_mask, (left_border, top_border),
             (right_border, bottom_border), 255, -1)

keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

print("BEFORE FILTER: " + str(len(keypoints_scene)))

# USE METHOD WITH CANNY FILTER
# Put true this flag if you want to use a filter which delete useless keypoints
filter_flag = True
if filter_flag:
    descriptors_scene, keypoints_scene = applyCannyFilter(
        img_scene, depth_image, sift_mask, keypoints_scene, descriptors_scene)

# Matching descriptor vectors with a FLANN based matcher
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

# Filter matches using the Lowe's ratio test
ratio_thresh = 0.72  # 0.72  # 0.87
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

print("GOOD MATCHES SIZE: " + str(len(good_matches)))

# Draw matches
img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]),
                       img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene,
               good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

oldSize = len(good_matches)
newSize = -1
j = 0

start = time.time()
# Sequential RANSAC
while((oldSize != newSize) and len(good_matches) >= 20):
    oldSize = len(good_matches)

    print("Iteration " + str(j))

    # Localize the object
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        # Get the keypoints from the good matches
        obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

    #print("Find homography")
    #H, mask = cv.findHomography(obj, scene, cv.RANSAC, confidence = 0.995, ransacReprojThreshold=4)
    #H, mask =  customFindHomography(obj, scene, 0.4)
    #H, mask =  customFindHomographyPlane3D(obj, scene, point_cloud, 0.55)
    #H, mask = customFindHomographyNormalSampling3D(obj, scene, point_cloud, 0.4, 0.1)
    #H, mask = customFindHomography3DTree(obj, scene, point_cloud, 0.4)
    H, mask = customFindHomographyDimension(
        obj, scene, point_cloud, 0.8, dimension)  # migliore è 0.6 o 1.0

    # H homography from template to scene
    H = np.asarray(H)
    # Take points from the scene that fits with the homography
    img_instance_matches = np.empty(
        (max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    maskk = (mask[:] == [0])
    instance_good_matches = ma.masked_array(
        good_matches, mask=maskk).compressed()

    # Remove inliers from good matches array
    new_good_matches = np.asarray(
        list(set(good_matches)-set(instance_good_matches)))
    good_matches = new_good_matches
    newSize = len(good_matches)

    # Show matches fitting the found homography
    #cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, instance_good_matches, img_instance_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imshow('Good Matches - 1 instance', img_instance_matches)
    #cv.imwrite("output_mask" + str(j) + ".jpg", img_instance_matches)
    # cv.waitKey()

    # Get the corners from the template
    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
    obj_corners[0, 0, 0] = 0
    obj_corners[0, 0, 1] = 0
    obj_corners[1, 0, 0] = img_object.shape[1]
    obj_corners[1, 0, 1] = 0
    obj_corners[2, 0, 0] = img_object.shape[1]
    obj_corners[2, 0, 1] = img_object.shape[0]
    obj_corners[3, 0, 0] = 0
    obj_corners[3, 0, 1] = img_object.shape[0]

    try:
        # Check for degenerate homography
        valid = True
        for k in range(0, len(obj_corners)):
            x = obj_corners[k, 0, 0]
            y = obj_corners[k, 0, 1]
            if (H[2][0]*x + H[2][1]*y + H[2][2]) / np.linalg.det(H) <= 0:
                valid = False

        # Draw lines between the corners
        scene_corners = cv.perspectiveTransform(obj_corners, H)

        # Draw Bounding box on the image
        if valid:
            end = time.time()
            print(
                "-------------------------------DRAWING BOX----------------------------------------")
            print("time elapsed:")
            print(end - start)

            cv.line(img_matches, (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])),
                    (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
            cv.line(img_matches, (int(scene_corners[1, 0, 0] + img_object.shape[1]), int(scene_corners[1, 0, 1])),
                    (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
            cv.line(img_matches, (int(scene_corners[2, 0, 0] + img_object.shape[1]), int(scene_corners[2, 0, 1])),
                    (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
            cv.line(img_matches, (int(scene_corners[3, 0, 0] + img_object.shape[1]), int(scene_corners[3, 0, 1])),
                    (int(scene_corners[0, 0, 0] + img_object.shape[1]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)

            #cv.imshow("prova", img_matches)
            # cv.waitKey()
            # cv.destroyAllWindows()

    except:
        print("Cannot draw bounding box")
        print(H)

    j += 1

 # End Sequential RANSAC

end = time.time()
print("time elapsed:")
print(end - start)

# Show detected matches
cv.imshow('Good Matches', img_matches)
#cv.imshow("AFTER FILTER", cdst)
#cv.imshow("ALL", cdst_copy)
cv.waitKey()
cv.destroyAllWindows()
#cv.imwrite("output_hough.jpg", cdst)
