import cv2

# Read an image
image = cv2.imread('E:/OpenCV/Images/cycle.jpg')

# Draw text on the image
cv2.putText(image, 'Hero Cycle', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 2, 255), 2, cv2.LINE_AA)
# cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

# Display the image with text
cv2.imshow('Text', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
