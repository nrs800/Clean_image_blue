import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image (grayscale)
image = cv2.imread('image001.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise before thresholding
blur = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Otsu's thresholding
_, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the results
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu Thresholding')
plt.axis('off')

plt.show()
