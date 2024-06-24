import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

src = cv.imread('C:/Users/skrua/source/repos/DLIP/Image/rice.png', 0)

cv.namedWindow('rice', cv.WINDOW_AUTOSIZE)
cv.imshow('rice', src)

blur = cv.blur(src,(5,5))

cv.namedWindow('blur', cv.WINDOW_AUTOSIZE)
cv.imshow('blur', blur)

# Structure Element for Morphology
cv.getStructuringElement(cv.MORPH_RECT,(5,5))
kernel = np.ones((5,5),np.uint8)
# Morphology
erosion = cv.erode(blur,kernel,iterations = 1)

cv.namedWindow('mor', cv.WINDOW_AUTOSIZE)
cv.imshow('mor', erosion)


cv.waitKey(0)

