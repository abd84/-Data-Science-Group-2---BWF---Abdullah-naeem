import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def apply_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

image_path = '/Users/abdullah/Desktop/Jupyter/tools-feature_blur-image_hero_mobile_2x.webp'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Define different convolutional kernels
# Sharpening kernel
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

# Smoothing (blurring) kernel
smoothing_kernel = np.array([[1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9]])

# Edge detection kernel (Sobel filter)
edge_kernel = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])

sharpened_image = apply_filter(image, sharpening_kernel)
smoothed_image = apply_filter(image, smoothing_kernel)
edge_image = apply_filter(image, edge_kernel)


plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(sharpened_image)
plt.title('Sharpened Image')

plt.subplot(1, 4, 3)
plt.imshow(smoothed_image)
plt.title('Smoothed Image')

plt.subplot(1, 4, 4)
plt.imshow(edge_image)
plt.title('Edge Detection')

plt.show()
