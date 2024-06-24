# * DLIP_Final Project_CNN Object Detection: Management of Expiration Dates and Inventory in Unmanned Stores
# * author: Gyeonheal An, TaegeonHan
# * Date: 2024-06-24
# * training code

from ultralytics import YOLO

def train():
    # Load a pretrained YOLO model
    model = YOLO('yolov8s.pt')

    # Train the model using the DLIP_Final.yaml dataset for epochs
    results = model.train(data="DLIP_Final.yaml", epochs=30)

if __name__ == '__main__':
    train()
