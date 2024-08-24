import numpy as np
import cv2
import serial
import time

# Load YOLOv3 model and configuration
net = cv2.dnn.readNet("F:\\Robotic ARM\\yolo\\yolov3.weights", "F:\\Robotic ARM\\yolo\\yolov3.cfg")
layer_names = net.getLayerNames()

# Load class names
classes = []
with open("F:\\Robotic ARM\\yolo\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configure output layers
output_layers = net.getUnconnectedOutLayersNames()

# Open a serial connection to Arduino
arduino = serial.Serial('COM20', 9600, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Set up window properties
window_name = "APPLE DETECTION WINDOW"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # Set window size

object_detected = False
command_sent = False
command_time = 0

ripe_apple_count = 0
unripe_apple_count = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize variables
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 47:  # Check if detected object is an apple
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Extract region of interest (ROI) for the apple
                roi = frame[max(center_y - h // 2, 0): min(center_y + h // 2, height),
                            max(center_x - w // 2, 0): min(center_x + w // 2, width)]

                # Convert ROI to HSV color space
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Calculate percentage of red and yellow-green pixels
                red_mask = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255])) + \
                           cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
                yellow_green_mask = cv2.inRange(hsv_roi, np.array([25, 52, 72]), np.array([86, 255, 255]))

                total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
                red_percentage = (np.count_nonzero(red_mask) / total_pixels) * 100
                yellow_green_percentage = (np.count_nonzero(yellow_green_mask) / total_pixels) * 100
                
                label = "Apple"
                if red_percentage > 5:
                    label += " (Ripe)"
                elif yellow_green_percentage > 5:
                    label += " (Unripe)"

                # Draw bounding boxes and labels
                font = cv2.FONT_HERSHEY_PLAIN
                color = (0, 255, 0)  # Green color for apple
                cv2.rectangle(frame, (max(center_x - w // 2, 0), max(center_y - h // 2, 0)),
                              (min(center_x + w // 2, width), min(center_y + h // 2, height)), color, 2)
                cv2.putText(frame, label, (max(center_x - w // 2, 0), max(center_y - h // 2, 0) + 30), font, 2, color, 2)

                # Determine if the apple is ripe or unripe and send command to Arduino
                if red_percentage > 5:
                    if not object_detected:
                        command_time = time.time()
                        object_detected = True

                    if object_detected and (time.time() - command_time) >= 5 and not command_sent:
                        arduino.write(b'1')
                        print("Command Sent: 1")
                        command_sent = True

                elif yellow_green_percentage > 5:
                    if not object_detected:
                        command_time = time.time()
                        object_detected = True

                    if object_detected and (time.time() - command_time) >= 5 and not command_sent:
                        arduino.write(b'2')
                        print("Command Sent: 2")
                        command_sent = True

    # Check for response from Arduino
    if command_sent:
        response = arduino.readline().decode('utf-8').strip()
        if response == 'Finished':
            print("Arduino response: Finished")
            if red_percentage > 5:
                ripe_apple_count += 1
            elif yellow_green_percentage > 5:
                unripe_apple_count += 1
            object_detected = False
            command_sent = False

    # Display output frame
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
arduino.close()
