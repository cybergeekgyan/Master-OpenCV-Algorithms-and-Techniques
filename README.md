# OpenCV-Techniques

Thresholding is a basic concept in computer vision. It is widely used in image processing.

Thresholding is used to simplify visual data for further analysis. In image processing, we need to pre-process the image data and get the important details. This technique is important to separate background image and foreground image.

# Thresholding

# Blurring and Smoothing


Images may contain lots of noise. There are few techniques through which we can reduce the amount of noise by blurring them. The blurring technique is used as a preprocessing step in various other algorithms. With blurring, we can hide the details when necessary. For example – the police use the blurring technique to hide the face of the criminal. Later on, when learning about edge detection in the image, we will see how blurring the image improves our edge detection experience.

There are different computer vision algorithms available to us which will blur the image using a kernel size. There is no right kernel size, according to our need, we have to try the trial and error method to see what works better in our case.

**There are four different types of functions available for blurring:**

**cv2.blur()** – This function takes average of all the pixels surrounding the filter. It is a simple and fast blurring technique.<br>
**cv2.GaussianBlur()** – Gaussian function is used as a filter to remove noise and reduce detail. It is used in graphics software and also as a preprocessing step in machine learning and deep learning models.<br>
**cv2.medianBlur()** – This function uses median of the neighbouring pixels. Widely used in digital image processing as under certain conditions, it can preserve some edges while removing noise.<br>
**cv2.bilateralFilter()** – In this method, sharp edges are preserved while the weak ones are discarded.<br>

# Color Filtering

When you need information about a specific color, you can take out the color you want.

This process is called **color filtering**. We can define a range of color which we want to focus at. For example – if you want to track a tennis ball in an image, we can filter out only green color from the image. This computer vision technique works better when the color that needs to be extracted is different from the background. 

*How to implement color filtering in your application.*

**Step 1: Convert the color image into HSV which is Hue Saturation Value**

**HSV** is also another way of representing a color. It is easier to specify a color range in HSV format that is why OpenCV expects us to specify the range in this format. Hue is for color (0- 179), Saturation is for the strength of color (0-255) and Value is for the different lighting conditions from low to high (0-255).

**Step 2: Create the mask**

OpenCV provides us with cv2.inRange() function. The first parameter contains the HSV image, the second parameter has a numpy array with lower bound of the color we want to extract and the third parameter contains the upper bound of the color.

**Note**: The upper and lower bound of color should be in HSV value. The mask contains only the 0 and 1 values representing black and white respective.

**Step 3: Perform bitwise ‘and’ operation**

At last, we perform bitwise ‘and’ operation on the original image and the mask image to get only the color we want.

# Edge Detection

Edges are a sudden change in the brightness of the image. The significant transitions in the brightness of the image are used to calculate the edges in an image. Edge detection is used for various image processing purposes. One of them is to sharpen the images. Sharpening of images is done to make the images more clear. Edge detection is used to enhance the images and image recognition becomes easier.

The canny edge detection algorithm is mostly used to detect the edges in an image.

OpenCV contains the function cv2.Canny() for edge detection.

cv2.Canny(img, 50, 150)

First parameter is the image data, the second parameter is the lower threshold value and the third parameter is the upper threshold value. You need to try different values for tuning the edge detection algorithm.
