import cv2
import mediapipe as mp

# Load the Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Replace with the correct path

def detect_faces(person):
    gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# media pipe

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

def detect_faces_mediapipe(image):
    # Convert the image color to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform face detection
    results = face_detection.process(rgb_image)
    # List to store bounding box coordinates
    faces = []

    if results.detections:
        for detection in results.detections:
            # Get the bounding box and convert it to pixel coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # Append the bounding box coordinates to the list
            faces.append((x, y, w, h))

    return faces