# * DLIP_Final Project_CNN Object Detection: Management of Expiration Dates and Inventory in Unmanned Stores
# * author: Gyeonheal An, TaegeonHan
# * Date: 2024-06-24
# * Main Code

from ultralytics import YOLO
import cv2 as cv
import time
import mediapipe as mp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Define expiry times for each class in seconds
expiry_times = {
    'triangle kimbap': 5,
    'Milk': 10,
    'coffee': 25,
    'Yogurt': 15,
    'Sandwich': 20
}

# Define prices for each class in Won
prices = {
    'triangle kimbap': 1600,
    'Milk': 1100,
    'coffee': 2100,
    'Yogurt': 2000,
    'Sandwich': 2500
}

# Store first detection times for each class
first_detection = {}

# Store last detection times for each class
last_detection = {}

# Store email sent status for each class
expired_email_flag = {}

# Store stock empty email sent status for each class
stock_email_flag = {}

# Define class names as per the YAML file
class_names = ['triangle kimbap', 'Milk', 'coffee', 'Yogurt', 'Sandwich']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Email configuration (Default)
smtp_server = "smtp.naver.com"
smtp_port = 587
# Email configuration (User)
smtp_user = ""              # Naver ID
smtp_password = ""          # Naver password
email_sender = ""           # Naver E-mail address
email_recipient = ""        # Receive E-mail address

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = email_recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(email_sender, email_recipient, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def process_frame(webcam, model):
    success, frame = webcam.read()
    if not success:
        return None, None, None

    results = model(frame)
    display_frame = results[0].plot()

    return frame, display_frame, results

def update_detection_times(results, current_time, display_frame): # Calculate Passed Time and Determine if the expiration date has passed
    detected_classes = set()
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = class_names[class_id]
        detected_classes.add(class_name)

        x1, y1, x2, y2 = map(int, result.xyxy[0])
        if class_name not in first_detection:                       # When it is detected first time --> Add the class
            first_detection[class_name] = current_time              # Indexing the first detection time for each classes
            expired_email_flag[class_name] = False                  # Initializing the flag
            stock_email_flag[class_name] = False
        last_detection[class_name] = current_time
        passed_time = current_time - first_detection[class_name]    # Calculating passed time

        if passed_time > expiry_times[class_name]:                  # When the product passed the expired date
            cv.putText(display_frame, f"{class_name} expired", (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
            if not expired_email_flag[class_name]:                  # Sending E-mail
                remaining_times = []                                # Store the remain times for product which not passed the expired date
                for other_class in class_names:                     # whole classes
                    if other_class in first_detection and other_class != class_name:    # A class other than the currently expired product class that has already been detected
                        remaining_time = expiry_times[other_class] - (current_time - first_detection[other_class])  # Calculating Remain time
                        remaining_times.append(f"{other_class}: {remaining_time:.2f} seconds remaining")
                msg_body = f"The {class_name} is expired.\n\nRemaining times for other items:\n" + "\n".join(remaining_times)   # Message
                send_email(f"{class_name} is expired", msg_body)
                expired_email_flag[class_name] = True

    check_stock_empty(detected_classes, current_time)

def check_stock_empty(detected_classes, current_time):
    for class_name in class_names:
        if class_name not in detected_classes:
            if class_name in last_detection:
                if current_time - last_detection[class_name] > 5 and not stock_email_flag.get(class_name, False):       # If it is not detected over programmed times
                    send_email(f"{class_name} is out of stock", f"The {class_name} is out of stock.")
                    stock_email_flag[class_name] = True             # E-mail flag not to send E-mail duplicate
            else:
                last_detection[class_name] = current_time           # Update the last detection time

    reset_detection_times(detected_classes, current_time)

def reset_detection_times(detected_classes, current_time):
    for class_name in list(first_detection.keys()):
        if class_name not in detected_classes:
            if current_time - last_detection[class_name] > 5:       # When it is not detected over programmed times
                first_detection.pop(class_name)                     # Clear all the product index and flags
                last_detection.pop(class_name)
                expired_email_flag.pop(class_name)
                stock_email_flag.pop(class_name, None)

def hand_detection(frame, display_frame, results):                # Hand detection function
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = display_frame.shape
            index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))        # coordinates of index finger

            for result in results[0].boxes:
                class_id = int(result.cls)
                class_name = class_names[class_id]
                x1, y1, x2, y2 = map(int, result.xyxy[0])

                if x1 < index_finger_tip_coords[0] < x2 and y1 < index_finger_tip_coords[1] < y2:       # If the finger tip is inside the contour box
                    price = prices[class_name]
                    cv.putText(display_frame, f"Price: {price} Won", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)

def main():
    model = YOLO('YOLOv8s_15.pt')

    webcam = cv.VideoCapture("test_video.mp4")
    # webcam = cv.VideoCapture(1)

    if not webcam.isOpened():
        print("Could not open webcam")
        return None

    while webcam.isOpened():
        frame, display_frame, results = process_frame(webcam, model)
        if frame is None:
            break

        current_time = time.time()                                      # Updating Current Time
        update_detection_times(results, current_time, display_frame)    # Expiration date management function
        hand_detection(frame, display_frame, results)

        cv.imshow("YOLOv8 Inference", display_frame)

        if cv.waitKey(3) & 0xFF == ord("q"):
            break

    webcam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
