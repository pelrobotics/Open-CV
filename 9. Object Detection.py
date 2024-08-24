import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("F:\\OpenCV\\yolo\\yolov3.weights", "F:\\OpenCV\\yolo\\yolov3.cfg")
layer_names = net.getLayerNames()
# Configure output layers
output_layers = net.getUnconnectedOutLayersNames()

# Load COCO names
with open("F:\\OpenCV\\yolo\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
image = cv2.imread("F:\\OpenCV\\Images\\apple.jpg")
height, width, channels = image.shape

# Prepare the image for the neural network
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialize lists to hold detected bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# For each detection from each output layer, get the confidence, class ID, and bounding box
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Set confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maxima Suppression to eliminate redundant overlapping boxes with lower confidences
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
