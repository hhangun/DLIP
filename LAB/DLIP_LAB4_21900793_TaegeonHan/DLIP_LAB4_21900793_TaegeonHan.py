"""
* author   :   Taegeon Han
* Date     :   2024-06-06
* Brief    :   DLIP_LAB: CNN Object Detection 1 (Parking Management System)
"""

import torch
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pandas


# Create counting_result.txt and enter a value
def store_count(value):
    with open('counting_result.txt', 'a') as file:
        file.write(value + '\n')

# Find available parking space number
def find_num(order):
    indexes = []
    for i, v in enumerate(order):
        if v == 0:
            indexes.append(i+1)
    return indexes

# Write the text on the display
def print_text(frame):
    cv.putText(frame,"Frame : " + f"{frame_count}",(20,585),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.putText(frame,"Number of cars : " + f"{car_count}",(20,625),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.putText(frame,"Number of available parking spaces : " + f"{park_space}",(20,665),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.putText(frame,"Available space number : " + f"{find_num(state_list)}",(20,705),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv.putText(frame,"Area Number: ", (5,230),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv.putText(frame,"1", (100,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"2", (213,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"3", (312,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"4", (412,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"5", (500,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"6", (588,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"7", (678,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"8", (761,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"9", (855,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"10", (941,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"11", (1031,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"12", (1133,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv.putText(frame,"13", (1237,265),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)


# Load the Model
model = YOLO('yolov8l.pt')

# # Check the class number of car
# class_list = model.names
# print(class_list)
car_num = 2

# Open the video camera
cap = cv.VideoCapture('DLIP_parking_test_video.avi')


# If not success, exit the program
if not cap.isOpened():
    print('Cannot open camera')
    exit()

# Define the region of interest (ROI)
roi = [[0, 255], [1280, 420]]
x1, y1 = roi[0]
x2, y2 = roi[1]

frame_count = 0

park_point = [110, 213, 312, 412, 500, 588, 678, 761, 855, 941, 1031, 1133, 1237]
park_len = len(park_point)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Crop the frame to the ROI
        roi_frame = frame[y1:y2, x1:x2]

        # Run YOLOv8 inference on the cropped frame
        results = model(roi_frame)

        # Initialize the car count
        car_count = 0

        state_list = [0 for no_use in range(park_len)]
        # Set only cars to come out
        for result in results:
            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            class_order = result.boxes.cls

            car_rect = boxes[class_order == car_num]
            car_confidence = scores[class_order == car_num]

            
            
            for box, score in zip(car_rect, car_confidence):
                if score >= 0.5:
                    xmin, ymin, xmax, ymax = box    # Allocate 4 coordinates
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)     # Convert to an integer to make a rectangle
                    cv.rectangle(roi_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Making state_list to 1 if park_point is located between xmin and xmax
                    for i in range(park_len):
                            if xmin < park_point[i] < xmax:
                                state_list[i] = 1

                    car_count += 1
            
            # If car_count is more than 13, available space is '0'
            if(car_count > park_len):
                park_space = 0

            # Calculate available parking spaces
            park_space = park_len - car_count

            # Print what we want
            print_text(frame)

            # Write the frame_count and car_count to the file
            counts = f"{frame_count},{car_count}"
            store_count(counts)

        #Resize the ROI to match the original region
        resize_ROI = cv.resize(roi_frame, (x2-x1, y2-y1))

        # Replace the ROI part in the original frame with the annotated ROI frame
        frame[y1:y2, x1:x2] = resize_ROI

        # Display the annotated frame
        cv.imshow("Video", frame)

        # Increase frame_count value by 1
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()