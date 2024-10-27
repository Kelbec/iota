import time
from ultralytics import YOLO
import cv2
import math 
import numpy as np
import argparse
import time
from collections import defaultdict
import screeninfo
from pythonosc import udp_client

timestep = 0.25  # seconds

def calculate_speed(coordinates, previous_position):
    # calculate speed
    speed = 0
    old_x, old_y = previous_position
    x1, y1, x2, y2 = coordinates

    new_x = (x1 + x2) / 2
    new_y = (y1 + y2) / 2
    position = (new_x, new_y)

    if old_x == 0 and old_y == 0:
        old_x = new_x
        old_y = new_y

    distance = ((new_x - old_x) ** 2 + (new_y - old_y) ** 2) ** timestep
    speed = distance / timestep
    previous_position = position

    return speed, previous_position

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
                    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=9001,
                    help="The port the OSC server is listening on")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)

# start webcam
cap = cv2.VideoCapture(0)
# Get screen resolution
screen = screeninfo.get_monitors()[0]  # Assuming you want to use the primary monitor
screen_width, screen_height = screen.width, screen.height

# Set video frame width and height to match screen resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Set the frame rate (in frames per second)
desired_frame_rate = 30.0
cap.set(cv2.CAP_PROP_FPS, desired_frame_rate)

# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person"]
positions = []
previous_position = (0, 0)

# Store the track history
track_history = defaultdict(lambda: [])

# Create a named window and set it to fullscreen
cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    results = model.track(img, stream=True)
    track_id = None

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
            center = (x1 + ((x2 - x1) / 2), y1 + ((y2 - y1) / 2))

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            if confidence > 0.65:
                # Draw bounding box
                r = cv2.rectangle(img, (x1, y1), (x2, y2), (50, 200, 25), 3)

                total_area = (x2 - x1) * (y2 - y1)
                speed, previous_position = calculate_speed([x1, x2, y1, y2], previous_position)
                delta = 50

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, f"person - {confidence}", org, font, fontScale, color, thickness)
                if box.id:
                    track_id = box.id.int().cpu().tolist()[0]
                    track = track_history[track_id]
                    track.append((float(previous_position[0]), float(previous_position[1])))
                    if len(track) > 30:
                        track.pop(0)
                    client.send_message("/object", [track_id, total_area / 1000, center[0], center[1], speed, confidence])

    # Display the image
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
