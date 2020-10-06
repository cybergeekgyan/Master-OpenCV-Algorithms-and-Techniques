import cv2
#load image
img = cv2.imread('car.jpeg')
#resizing image
img = cv2.resize(img, (420,300))
#applying canny edge detection algorithm
canny = cv2.Canny(img,50,150)
#displaying results
cv2.imshow('image',img)
cv2.imshow('canny',canny)
#destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
