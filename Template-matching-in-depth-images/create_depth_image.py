import cv2 as cv
import numpy as np
import numpy.ma as ma
import math

point_cloud = np.load("3D/4/pointCloud.npy")  # [height][width][xyz]

height = 480
width = 640

depth_image = np.empty((height, width), dtype=np.float32)

# Trova la profondità massima
maximum = 0
for h in range(height):
    for w in range(width):
        if(point_cloud[h][w][2] > maximum):
            maximum = point_cloud[h][w][2]
# print(maximum)

# Trova la profondità minima --> è sempre zero in realtà
minimum = 0
for h in range(height):
    for w in range(width):
        if(point_cloud[h][w][2] < minimum):
            minimum = point_cloud[h][w][2]
# print(minimum)


for h in range(height):
    for w in range(width):
        depth_image[h][w] = (point_cloud[h][w][2] - minimum) * \
            255.0 / (maximum - minimum)
        #depth_image[h][w] = point_cloud[h][w][2]


cv.imshow('Depth Image', depth_image)
#cv.imwrite('3D/2/depth_image.jpg', depth_image)
cv.waitKey()
cv.destroyAllWindows()
