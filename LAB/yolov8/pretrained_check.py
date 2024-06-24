from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import torch.nn as nn



# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')


# Check the class number of car
class_list = model.names
print(class_list)