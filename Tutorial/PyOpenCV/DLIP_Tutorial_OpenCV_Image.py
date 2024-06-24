import cv2 as cv

# Read image
HGU_logo = 'C:\\Users\\skrua\\source\\repos\\DLIP\\Image\\HGU_logo.jpg'
src = cv.imread(HGU_logo)
src_gray = cv.imread(HGU_logo, cv.IMREAD_GRAYSCALE)

# Write image
HGU_logo_name = 'writeImage.jpg'
cv.imwrite(HGU_logo_name, src)

# Display image
cv.namedWindow('src', cv.WINDOW_AUTOSIZE)
cv.imshow('src', src)

cv.namedWindow('src_gray', cv.WINDOW_AUTOSIZE)
cv.imshow('src_gray', src_gray)

cv.waitKey(0)