import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load image
img = cv.imread('C:/Users/skrua/source/repos/DLIP/Image/EdgeLineImages/coins.png',cv.COLOR_BGR2GRAY)

# Apply Canndy Edge
edges = cv.Canny(img,50,200)

# Plot Results
#cv.imshow('Edges',edges)
titles = ['Original','Edges']
images = [img, edges]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

#################################################################################33
# Load image
src = cv.imread('C:/Users/skrua/source/repos/DLIP/Image/EdgeLineImages/coins.png',cv.COLOR_BGR2GRAY)
# Apply Thresholding then Canndy Edge
thVal=127
ret,thresh1 = cv.threshold(src,thVal,255,cv.THRESH_BINARY)
edges2 = cv.Canny(thresh1,50,200)

# Plot Results
#cv.imshow('Edges',edges)
titles = ['Original','Edges']
images = [src, edges2]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
    