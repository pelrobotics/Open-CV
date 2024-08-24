import cv2

# Read an image
image = cv2.imread('F:/OpenCV/Images/cycle.jpg')


# Draw a rectangle
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 10)
# (image, start_point, end_point, color, thickness)
#start_point(x,y) end_point(x,y) color (B,G,R) thickness (3)

# Draw a circle
cv2.circle(image, (300, 300), 50, (255, 0, 0), -1)
# (image, center, radius, color, thickness)

# Draw a line
cv2.line(image, (100, 100), (300, 300), (0, 0, 255), 5)
# (image, start_point, end_point, color, thickness)

# Display the image with shapes
cv2.imshow('Shapes', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
