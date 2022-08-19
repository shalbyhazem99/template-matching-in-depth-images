import math
import cv2 as cv
import numpy as np

# Determina l'angolo tra due rette e ritorna un booleano a seconda della threshold


def findTheta(x1, y1, x2, y2,
              x3, y3, x4, y4,
              theta_thresh):
    # center around 0
    h1 = y2 - y1
    w1 = x2 - x1
    h2 = y4 - y3
    w2 = x4 - x3
    # normalize to unit vectors
    h1u = h1 / np.sqrt(pow(h1, 2) + pow(w1, 2))
    w1u = w1 / np.sqrt(pow(h1, 2) + pow(w1, 2))
    h2u = h2 / np.sqrt(pow(h2, 2) + pow(w2, 2))
    w2u = w2 / np.sqrt(pow(h2, 2) + pow(w2, 2))

    cos_theta = h1u*h2u + w1u*w2
    return abs(cos_theta) < theta_thresh


src = cv.imread("3D/5/depth_image.jpg")
grey = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src2 = np.copy(src)

depth_denoised = cv.fastNlMeansDenoising(grey, None, 80, 7, 21)
u8_denoised = (depth_denoised*255).astype(np.uint8)

dst = cv.Canny(u8_denoised, 100, 170, None, 3)

ret, thresh = cv.threshold(dst, 250, 255, cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(src2, contours, -1, (0, 255, 0), 2)

cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

lines = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 40, 40)

if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv.line(cdst, (l[0], l[1]), (l[2], l[3]),
                (0, 0, 255), 2, cv.LINE_AA)

cv.imshow("Source", src)
cv.imshow("Source with counters", src2)
cv.imshow("Canny", dst)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdst)

cv.waitKey()
cv.destroyAllWindows()
