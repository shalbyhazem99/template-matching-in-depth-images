import cv2
import numpy as np

# Load image, bilaterial blur, and Otsu's threshold
image = cv2.imread("3D/5/rgb_image.jpg")
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray,9,75,75)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform morpholgical operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)


# Draw rectangles
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('close', close)
cv2.imshow('image', image)
cv2.waitKey()