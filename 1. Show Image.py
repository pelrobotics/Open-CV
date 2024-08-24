#pip install opencv-python

#import serial

import cv2

# Read an image

#image -- variable 
image = cv2.imread('F:/OpenCV/Images/cycle.jpg') #F:/OpenCV/Images/cycle.jpg  |  / -- open the file



# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the image in a window
cv2.imshow('Orignal Image', image)
cv2.imshow('Gray Scale Image', gray_image) 


#cv2.imwrite('F:/OpenCV/Images/cycle.jpg', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
