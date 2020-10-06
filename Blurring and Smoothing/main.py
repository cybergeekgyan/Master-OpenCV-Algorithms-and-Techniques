import cv2
#Reading image from disk
dog = cv2.imread('dog.jpg')
#Resizing image to make it smaller
dog = cv2.resize(dog, (300,280) )
#Applying different blur functions with 7*7 filter
img_0 = cv2.blur(dog, ksize = (7, 7))
img_1 = cv2.GaussianBlur(dog, (7, 7), 0)
img_2 = cv2.medianBlur(dog, 7)
img_3 = cv2.bilateralFilter(dog, 7, 75, 75)
#Displaying resultant images
cv2.imshow('Original', dog)
cv2.imshow('Blur', img_0)
cv2.imshow('Gaussian Blur', img_1)
cv2.imshow('Median Blur', img_2)
cv2.imshow('Bilateral Filter', img_3)
#Waits for a user to press key for exit
cv2.waitKey(0)
cv2.destroyAllWindows()
