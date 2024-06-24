"""
* DLIP_LAB3.py
* author: Taegeon Han
* Date: 2024-05-06
* DLIP_LAB3_Tension Detection of Rolling Metal Sheet
* Measure the metal sheet tension level
"""

import cv2 as cv
import numpy as np


# Challenge picture
image_width = 1920
image_height = 1080
src = cv.imread('Challenge_LV1.png')
src = cv.resize(src, (image_width,image_height))

# Create a new image highlighting the details of the red channel
B, G, R = cv.split(src)
R_enhanced = cv.equalizeHist(R)
result_image = cv.merge((B, G, R_enhanced))

# Draw the line to divide level
cv.line(result_image, (0,result_image.shape[0]-120), (result_image.shape[1],result_image.shape[0]-120), (0,255,0), 2, cv.LINE_AA)
cv.line(result_image, (0,result_image.shape[0]-250), (result_image.shape[1],result_image.shape[0]-250), (255,0,0), 2, cv.LINE_AA)
dst = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)

while  cv.waitKey(1) != ord('q'):
    
    # Apply blur
    median = dst 
    for i in range(0,1):
        median = cv.medianBlur(median,9)
    for i in range(0,1):
        median = cv.GaussianBlur(median, (7, 7), 0)

    # Apply Thresholding
    thVal = 49
    ret,thresholding = cv.threshold(median,thVal,255,cv.THRESH_BINARY)

    # Apply Morphology
    size = 21
    size2 = 15
    kernel = np.ones((size, size),np.uint8)
    kernel2 = np.ones((size2, size2),np.uint8)

    erosion = cv.erode(thresholding,kernel,iterations = 1)
    opening = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel)
    opening = cv.dilate(opening, kernel2, iterations = 1)

    # Masking Image
    mask = np.zeros(dst.shape, dtype="uint8")
    cv.rectangle(mask, (0,440), (800,1080), 255, -1)
    mask2 = np.zeros(dst.shape, dtype="uint8")+255
    cv.rectangle(mask2, (618,630), (1920,1080), 255, -1)
    mask_image = cv.bitwise_and(opening, opening, mask=mask)
    #mask_image = cv.bitwise_and(mask_image2, mask_image2, mask=mask2)


    x_limit, y_limit = mask_image.shape[1], mask_image.shape[0]  # Length and width of mask_image
    last_x, last_y = -1, -1     # Store the x, y coordinates of previously discovered pixels
    y_dist = 12     # Allowable pixel distance in y-direction

    low_circle = 1  # Bottom Circle

    # Check all the pixels in the image
    for x in range(x_limit):
        for y in range(y_limit):
            if mask_image[y, x] == 255:  # Check white pixel in mask_image
                if last_y == -1 or abs(last_y - y) <= y_dist and abs(last_x - x) <= 3:  # Distance conditions from previous pixels
                    cv.circle(result_image, (x, y), 1, (0, 255, 0), -2)  # Draw circle
                    last_x = x
                    last_y = y

                    # Find the bottom circle
                    if low_circle < last_y:
                        low_circle = last_y
                    else:
                        low_circle = low_circle


    # resize the picture
    resize_width = 600
    resize_height = 400
    result_image = cv.resize(result_image, (600, 400))

    # Calculate by the changed size
    low_circle = low_circle*(resize_height/image_height)

    score = (low_circle / resize_height) * 100  
    location_score = (int(resize_width-resize_width/3), int(resize_height/4))
    location_level = (int(resize_width-resize_width/3), int(resize_height/4 + 30))

    # Show the score, level in the picture
    if low_circle < result_image.shape[0]-250*(resize_height/image_height): # Level 1
        cv.putText(result_image, f"Score: {score:.2f}", location_score, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.putText(result_image, f"Level: 1", location_level, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        mask_image = cv.bitwise_and(mask_image, mask_image, mask=mask2)

    elif low_circle >= result_image.shape[0]-120*(resize_height/image_height): # Level 3
        cv.putText(result_image, f"Score: {score:.2f}", location_score, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.putText(result_image, f"Level: 3", location_level, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:   # Level 2
        cv.putText(result_image, f"Score: {score:.2f}", location_score, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.putText(result_image, f"Level: 2", location_level, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

   
    cv.imshow("Contours", result_image)

    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
    

cv.waitKey(0)
cv.destroyAllWindows()