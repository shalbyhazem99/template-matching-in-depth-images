import cv2 as cv
import numpy as np
import argparse
import numpy.ma as ma
import os
import random
from customRansac import customFindHomography
from Template import Template

# Detect the keypoints using SIFT/SURF Detector, compute the descriptors
minHessian = 400
#detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
detector = cv.SIFT_create()

# Read all templates
template_files = os.listdir("Templates")
templates = []
for f in template_files:
    templates.append(Template("Templates/" + str(f)))

img_scene = cv.imread("3D/1/rgb_image.jpg")

# Compute keypoints and descriptors
keypoints_templates = []
descriptors_templates = []
for i, t in enumerate(templates):
    keypoints_obj, descriptors_obj = detector.detectAndCompute(t.image, None)
    keypoints_templates.append(keypoints_obj)
    descriptors_templates.append(descriptors_obj)
    detector = cv.SIFT_create()

keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

# Matching descriptor vectors with a FLANN based matcher for each template
good_matches_all = []
for i, t in enumerate(templates):
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(
        descriptors_templates[i], descriptors_scene, 2)

    # Filter matches using the Lowe's ratio test
    ratio_thresh = 0.8
    good_matches_obj = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches_obj.append(m)
    good_matches_all.append(good_matches_obj)

# Draw matches on the image
img_matches = img_scene

for i, t in enumerate(templates):

    # stop flags
    good_matches = good_matches_all[i]
    oldSize = len(good_matches)
    newSize = -1
    j = 0

    # Random color
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = (r, g, b)

    # Sequential RANSAC
    while(newSize != oldSize and len(good_matches) >= 4):
        oldSize = len(good_matches)

        # Localize the object
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)
        for z in range(len(good_matches)):
            # Get the keypoints from the good matches
            obj[z, 0] = (keypoints_templates[i])[
                good_matches[z].queryIdx].pt[0]
            obj[z, 1] = (keypoints_templates[i])[
                good_matches[z].queryIdx].pt[1]
            scene[z, 0] = keypoints_scene[good_matches[z].trainIdx].pt[0]
            scene[z, 1] = keypoints_scene[good_matches[z].trainIdx].pt[1]
        #cv.findHomography(obj, scene, cv.RANSAC, confidence = 0.995, ransacReprojThreshold=5)
        H, inliers_mask = customFindHomography(obj, scene, 0.7)
        # H homography from template to scene
        H = np.asarray(H)
        # Take points from the scene that fits with the homography
        mask = (inliers_mask[:] == [0])
        instance_good_matches = ma.masked_array(
            good_matches, mask=mask).compressed()

        # Get the corners from the template
        obj_corners = np.empty((4, 1, 2), dtype=np.float32)
        obj_corners[0, 0, 0] = 0
        obj_corners[0, 0, 1] = 0
        obj_corners[1, 0, 0] = templates[i].image.shape[1]
        obj_corners[1, 0, 1] = 0
        obj_corners[2, 0, 0] = templates[i].image.shape[1]
        obj_corners[2, 0, 1] = templates[i].image.shape[0]
        obj_corners[3, 0, 0] = 0
        obj_corners[3, 0, 1] = templates[i].image.shape[0]

        try:
            if len(instance_good_matches) > 4:
                # Check for degenerate homography
                valid = True
                for k in range(0, len(obj_corners)):
                    det = np.linalg.det(H)
                    if det != 0:
                        x = obj_corners[k, 0, 0]
                        y = obj_corners[k, 0, 1]
                        if (H[2][0]*x + H[2][1]*y + H[2][2]) / det <= 0:
                            valid = False
                    else:
                        valid = False

                if(valid):
                    # Remove inliers from good matches array
                    new_good_matches = np.asarray(
                        list(set(good_matches)-set(instance_good_matches)))
                    good_matches = new_good_matches
                    newSize = len(good_matches)

                    # Draw lines between the corners
                    scene_corners = cv.perspectiveTransform(obj_corners, H)
                    cv.line(img_matches, (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])),
                            (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), color, 4)
                    cv.line(img_matches, (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])),
                            (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), color, 4)
                    cv.line(img_matches, (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])),
                            (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), color, 4)
                    cv.line(img_matches, (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])),
                            (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), color, 4)
                else:
                    random.shuffle(good_matches)
                    newSize = len(good_matches)

            else:
                random.shuffle(good_matches)
                newSize = len(good_matches)
        except:
            print("Cannot draw bounding box")
            print(H)

     # End Sequential RANSAC

# Show detected matches
cv.imwrite("output.jpg", img_matches)
