import cv2
#Reading image from computer
img = cv2.imread('free.jpg')
#Resizing image to make it smaller
img = cv2.resize(img, (320,240))
#Converting color image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Applying threshold to the gray image
ret, thresh1 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray_img, 125, 255, cv2.THRESH_TOZERO_INV)
#Displaying resultant images
cv2.imshow('Original Image', img)
cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)
#Wait for user to press key for exit
cv2.waitKey(0)
cv2.destroyAllWindows()
