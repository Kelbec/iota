import time
from ultralytics import YOLO
import cv2
import math 
import numpy as np
import argparse
import time
from collections import defaultdict

from pythonosc import udp_client

# TODO: Boost CPU performance with openvino model https://medium.com/@gkhndmn4/yolo-cpu-performance-enhancement-3685b7fa84a5

timestep = 0.5  # seconds


def calculate_speed(coordinates,previous_position):
    # calculate speed
    speed = 0
    old_x,old_y = previous_position
    x1,y1,x2,y2 = coordinates

    new_x = (x1+x2)/2
    new_y = (y1+y2)/2
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
cap = cv2.VideoCapture(1)
cap.set(3,1200)# 1920)
cap.set(4,600)# 1080)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person"]
            #   , "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            #   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            #   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            #   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            #   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            #   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            #   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            #   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            #   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            #   "teddy bear", "hair drier", "toothbrush"
            #   ]
positions = []
previous_position = (0,0)

# Store the track history
track_history = defaultdict(lambda: [])


while True:
    success, img = cap.read()
    # results = model(img, stream=True)
    results = model.track(img, stream=True)
    track_id = None
    # coordinates
    for r in results:
        
        boxes = r.boxes
        
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            center = (x2-x1, y2-y1)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            if confidence > 0.65:
                # print("Confidence --->",confidence)
                # print("ID --->",track_id)
                # put box in cam
                r = cv2.rectangle(img, (x1, y1), (x2, y2), (50, 200, 25), 3)
                
                # print("Coordinates of the rectangle:", x1, y1, x2, y2)
                total_area = (x2-x1) * (y2-y1)
                # print("Total area:",total_area)

                # print("previous0",previous_position)
                speed, previous_position = calculate_speed([x1,x2,y1,y2],previous_position)
                # print("previous1",previous_position)
                # print("SPEED:",speed)
                delta = 50
                # Extract the region of interest (ROI) within the rectangle
                roi = img[y1+delta:y1+y2-delta, x1+delta:x1+x2-delta]

                # debugging rectangle
                # cv2.rectangle(img, (x1+delta, y1+delta), (x2-delta, y2-delta), (200, 0, 25), 3)

                # Calculate the average color values within the ROI
                # average_color = np.mean(roi, axis=(0, 1))

                # Print the average color values in RGB
                # print("Average color values in RGB:", average_color)
                

                # # class name
                # cls = int(box.cls[0])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, f"person - {confidence}", org, font, fontScale, color, thickness)
                # data = boxes.data.cpu().tolist()
                # print("DATA")
                # print(data)
                if box.id:
                    track_id = box.id.int().cpu().tolist()[0]   # int(row[-3])  # track ID
                    track = track_history[track_id]
                    track.append((float(previous_position[0]), float(previous_position[1])))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)
                    print("TRACK ID ",track_id)
                # for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
                #     if boxes.is_track:
                        
                #         print("ID")
                #         print(track_id)
                #         # print(center)
                #         if track_id:
                    client.send_message("/object", [track_id, total_area/1000, *center])#*average_color, total_area])

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    # print(positions)
    time.sleep(timestep)

cap.release()
cv2.destroyAllWindows()
