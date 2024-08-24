#pip install numpy==1.26.2

import cv2
import dlib 
import numpy as np

# Load face recognition models from dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("F:\\OpenCV\\Dlib\\shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("F:\\OpenCV\\Dlib\\dlib_face_recognition_resnet_model_v1.dat")

# Load images and corresponding names
m1_image = cv2.imread("F:\\OpenCV\\Images\\m1.jpg")
m2_image = cv2.imread("F:\\OpenCV\\Images\\m2.jpg")
m3_image = cv2.imread("F:\\OpenCV\\Images\\m3.jpg")
# Convert images to RGB format (dlib uses RGB)
m1_image_rgb = cv2.cvtColor(m1_image, cv2.COLOR_BGR2RGB)
m2_image_rgb = cv2.cvtColor(m2_image, cv2.COLOR_BGR2RGB)
m3_image_rgb = cv2.cvtColor(m3_image, cv2.COLOR_BGR2RGB)
# Detect face landmarks and compute face encodings
m1_face = detector(m1_image_rgb)[0]
m1_shape = shape_predictor(m1_image_rgb, m1_face)
m1_face_encoding = face_recognizer.compute_face_descriptor(m1_image_rgb, m1_shape)

m2_face = detector(m2_image_rgb)[0]
m2_shape = shape_predictor(m2_image_rgb, m2_face)
m2_face_encoding = face_recognizer.compute_face_descriptor(m2_image_rgb, m2_shape)

m3_face = detector(m3_image_rgb)[0]
m3_shape = shape_predictor(m3_image_rgb, m3_face)
m3_face_encoding = face_recognizer.compute_face_descriptor(m3_image_rgb, m3_shape)



known_face_encodings = [m1_face_encoding, m2_face_encoding, m3_face_encoding]
known_face_names = ["M1", "M2", "M3"]

# Open video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Unable to read frame from the camera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(frame_rgb)

    for face in faces:
        shape = shape_predictor(frame_rgb, face)
        face_encoding = face_recognizer.compute_face_descriptor(frame_rgb, shape)

        matches = [np.linalg.norm(np.array(face_encoding) - np.array(known_encoding)) < 0.6 for known_encoding in known_face_encodings]
        name = "Unknown"

        if True in matches:
            name = known_face_names[matches.index(True)]

        top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(30) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
