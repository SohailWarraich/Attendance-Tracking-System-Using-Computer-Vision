#Employee Attendance System - v1

This is the initial version (v1) of the Employee Attendance System, which uses computer vision for employee tracking and face detection.

##Features
YOLO Detection: Utilizes the YOLO model to detect people and vehicles (class IDs 0 and 2).

Function: YOLO_Detection(model, frame, conf=0.35)
Output: Bounding boxes, classes, names, and IDs.
##Face Detection:

Haar Cascade: Uses Haar Cascade Classifier to detect faces in grayscale images.
Function: detect_faces(person)
MediaPipe: Uses MediaPipe for more advanced face detection.
Function: detect_faces_mediapipe(image)
Region of Interest (ROI) Selection:

##File: get_roi_points.py
Function: Allows users to select ROI points in the video feed where face recognition or detection will occur, focusing the attendance system on specific areas.
Attendance Capture:

In this version, the system captures an employee's face when they approach the specified region of interest (ROI).
The detected face is displayed on the screen, preparing the system for future upgrades to save faces and track identities.
Planned Feature: In later versions, these faces or their identities can be saved in a database for attendance record-keeping.

##Setup
Install the required libraries:

`pip install opencv-python mediapipe
Ensure you have the YOLO model weights and haarcascade_frontalface_default.xml for face detection with Haar Cascade.

Place utils_employee.py and get_roi_points.py in the same directory.

##Usage
Run the EmployeeAttendance.py script to initiate the attendance tracking system.
Import YOLO_Detection, detect_faces, or detect_faces_mediapipe for face detection functionalities.
Use get_roi_points.py to define specific regions for face detection and recognition.

##Files
EmployeeAttendance.py: Main script for running the attendance system.
utils_employee.py: Utility file with additional functions for processing.
get_roi_points.py: Script to select ROI points for focused face detection and recognition.

