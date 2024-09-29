import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (assuming it's a binary black and white image)
img = cv2.imread('/home/nathanael-seay/Documents/Otsu image cropped.png', cv2.IMREAD_GRAYSCALE)

# Apply threshold if needed (to ensure binary image)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find edges using Canny edge detection
edges = cv2.Canny(binary, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw the contours on the original image
contour_img = np.zeros_like(binary)
cv2.drawContours(contour_img, contours, -1, (255), 1)

# Display the image with contours
plt.imshow(contour_img, cmap='gray')
plt.title('Traced Path')
plt.show()
