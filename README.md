# OpenCV-Techniques

Thresholding is a basic concept in computer vision. It is widely used in image processing.

Thresholding is used to simplify visual data for further analysis. In image processing, we need to pre-process the image data and get the important details. This technique is important to separate background image and foreground image.

In the starting, our image is colorful and it will contain 3 values ranging from 0-255 for every pixel. In thresholding, first, we have to convert the image in gray-scale. So for a gray-scale image, we only need 1 value for every pixel ranging from 0-255. By converting the image, we have reduced the data and we still have the information.

The next step in thresholding is to define a threshold value that will filter out the information which we don’t want. All the pixel values less than the threshold value will become zero and the pixel values greater than the threshold value will become 1. As you can see, now our image data only consist of values 0 and 1.

The function used for the threshold is given below:

cv2.threshold(img , 125, 255, cv2.THRESH_BINARY)

The first parameter is the image data, the second one is the threshold value, the third is the maximum value (generally 255) and the fourth one is the threshold technique. Let’s implement this in our code and observe how different techniques affect the image.
