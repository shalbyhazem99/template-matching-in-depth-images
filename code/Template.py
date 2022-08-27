import cv2 as cv


class Template:

    def __init__(self, filename):
        self.filename = filename
        self.name = filename
        self.image = cv.imread(filename)
