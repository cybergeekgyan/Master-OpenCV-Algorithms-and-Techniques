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

#
