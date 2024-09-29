import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image (grayscale)
image = cv2.imread('image001.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise (optional)
blur = cv2.GaussianBlur(image, (5, 5), 0)

# Manually set the threshold value
threshold_value = 95 # You can adjust this value
_, manual_thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

# Display the results
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(manual_thresh, cmap='gray')
plt.title(f'Manual Thresholding (Threshold = {threshold_value})')
plt.axis('off')

plt.show()

#----------------------------------------------------------------------

plt.imshow(manual_thresh, cmap='gray')

plt.axis('off')

plt.show()