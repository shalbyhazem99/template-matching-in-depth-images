import cv2
import numpy as np
import numpy.ma as ma
import math


img_scene = cv2.imread("3D/5/rgb_image.jpg")

template = "barchette.png"
template_path = "Templates/" + template
img_object = cv2.imread(template_path)

obj_gray = cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)

avg_color_per_row = np.average(obj_gray, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print(avg_color)


img_gray = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)


# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, avg_color, 255, cv2.THRESH_BINARY_INV)
#th3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

# visualize the binary image
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                     
# draw contours on the original image
image_copy = img_scene.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
               
# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
#cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()


