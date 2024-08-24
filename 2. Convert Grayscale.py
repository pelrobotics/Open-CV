import cv2

# Read an image
image = cv2.imread('E:/OpenCV/Images/cycle.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
