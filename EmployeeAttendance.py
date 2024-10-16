import glob
import torch
import cv2
import numpy as np
import datetime
import os
import time
from utils_employee import detect_faces, detect_faces_mediapipe, YOLO_Detection
from ultralytics import YOLO

# Yolov5s Model for Person Detection
# model = torch.hub.load('ultralytics/yolov5', 'yolov8')
model = YOLO("yolov8n")
model.classes = [0]
model.conf = 0.85

# Initialize VidepCapture and get video FPS
cap = cv2.VideoCapture("input_video/employee.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define Region of Interest Coordinates
roi_coords = [(57, 508), (52, 638), (300, 639), (287, 515)]
# roi_coords2 = [(529, 501), (930, 524), (914, 703), (368, 645)]

# Counter number of detections inside ROI
detections_counter = 0
# Delay Counter to ignore idle persons walk from ROI
delayCounter = 0
# Tracker if no more detection inside ROI, it reset delayCounter
resetCounter = []

# To write output video in mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file format
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
face_count = 0
idx = 1
last_save_time = time.time()
while True:
    ret, frame = cap.read()

    if not ret:
        break
    cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (100, 150, 30), 3, lineType=cv2.LINE_AA)

    # Folder where face images stored
    face_images = glob.glob("output/*.jpg")

    # If Counter
    resetCounter.append(detections_counter)
    if len(resetCounter) > 2:
        resetCounter.pop(0)

    # Get the current time in "H:M AM/PM" format Draw Banner on Top of Frame
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p")    
    cv2.rectangle(frame, (0, 0), (380, 30), (0,120, 110), -1)  # Rectangle dimensions and color
    # Put the current time on top of the black rectangle
    cv2.putText(frame, f"Attendance Monitoring: {current_time}", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (190, 215, 255), 1, cv2.LINE_AA)

    try:
        # Perform YOLO detection
        boxes, classes, names, ids = YOLO_Detection(model, frame)

        # Collect points to determine if any detection is inside polygons
        detection_points = []
        for box, id in zip(boxes, ids):
            id = int(id)
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Calculate the center point of the bounding box
            center_x = ((x1 + x2) / 2)
            center_y = y2 - 40
            center_point = (int(center_x), int(center_y))
            detection_points.append((int(center_x), int(center_y)))

            # Define the color of the circle (BGR format)
            circle_color = (0, 120, 0)  # Green color in BGR

            result = cv2.pointPolygonTest(np.array(roi_coords, dtype=np.int32), center_point, False)
            idx = id
            if result > 0 and idx == id:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 120, 255), 1)
                detections_counter += 1
                # if delayCounter == 10 or delayCounter == 49 or delayCounter == 173:
                if delayCounter == 6 or delayCounter == 10:
                    # Extract the person's bounding box
                    person = frame[y1:y2, x1:x2]
                    # Convert the person bounding box to grayscale for face detection
                    gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
                    # Perform face detection within the person bounding box using Haar Cascade
                    # faces = detect_faces(person)
                    faces = detect_faces_mediapipe(person)
                    for (fx, fy, fw, fh) in faces:
                        if time.time() - last_save_time >= 1:
                            # Adjust the face coordinates based on the person's bounding box
                            fx1, fy1, fx2, fy2 = x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh
                            # Extract the face
                            face = frame[fy1 - 5:fy2 + 5, fx1 - 5:fx2 + 5]

                            # Save each face with a unique name
                            face_filename = f'output/face_{face_count}.jpg'
                            cv2.imwrite(face_filename, face)
                            face_count += 1  # Increment the counter for unique filenames

                            # Update the last save time to the current time
                            last_save_time = time.time()


                if len(face_images) > 0:
                    y_offset = 40
                    image = cv2.imread(face_images[-1])
                    height, width, _ = image.shape
                    frame[y_offset:y_offset + height, 40:width + 40] = image
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                # idx = id
                # idx = id + 1

            if len(resetCounter) > 1:
                if resetCounter[1] - resetCounter[0] == 0:
                    delayCounter = 0
                    try:
                        if len(face_images) > 0:
                            os.remove(face_images[-1])
                    except: pass
                else:
                    delayCounter += 1
    except:
        pass

    out.write(frame)
    cv2.imshow("Camera-1", frame)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()