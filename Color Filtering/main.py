import cv2
import numpy as np
#Reading images from local disk
img = cv2.imread('ball.jpeg')
#Resizing image to make it smaller(optional)
img = cv2.resize(img, (320,240))
#Defining lower and upper range for color we want to extract(hsv format)
lower_orange = np.array([8, 100, 50])
upper_orange = np.array([15, 255, 255])
#Converting image from BGR to HSV format
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#cv2.inRange() function gives us the mask having colors within the specified lower and upper range
mask = cv2.inRange(hsv_img, lower_orange, upper_orange)
#cv2.bitwise_and() function performs bitwise and operation on pixels of original image and mask
res = cv2.bitwise_and(img, img, mask=mask)
#Displaying all the output images
cv2.imshow('Original',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
#Waiting for user to press key for exit
cv2.waitKey(0)
cv2.destroyAllWindows()
