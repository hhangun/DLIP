from ultralytics import YOLO
import cv2

def load_video(video_path):
    return cv2.VideoCapture(video_path)
def test():

    # Load a pretrained YOLO model(Change model directory)
    model = YOLO('runs/detect/train/weights/best.pt')
    video_path = '' # Video path
    # Open the video camera no.0
    cap = load_video(video_path)

    #cap = cv2.VideoCapture(0)

    # If not success, exit the program
    if not cap.isOpened():
        print('Cannot open camera')

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()