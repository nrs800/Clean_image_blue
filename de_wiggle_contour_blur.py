import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Function to smooth contours using Gaussian smoothing
def smooth_contour(contour, sigma=0.5):
    # Convert the contour to an array of points
    contour = np.squeeze(contour)

    # Check if contour has enough points
    if len(contour) < 3:
        return contour  # Can't smooth with very few points

    # Smooth x and y coordinates separately using gaussian_filter1d
    x = gaussian_filter1d(contour[:, 0], sigma)
    y = gaussian_filter1d(contour[:, 1], sigma)

    # Stack them back together
    smoothed_contour = np.vstack((x, y)).T
    return smoothed_contour

# Load the image and find contours
img = cv2.imread('/home/nathanael-seay/Documents/Otsu image cropped.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Resample the contour (Optional)
epsilon = .01 * cv2.arcLength(contours[0], True)  # Resampling based on 1% of contour arc length
resampled_contour = cv2.approxPolyDP(contours[0], epsilon, True)

# Smooth the resampled contour
smoothed_contour = smooth_contour(resampled_contour)

# Draw the smoothed contour on a blank image
contour_img = np.zeros_like(binary)
smoothed_contour = smoothed_contour.astype(np.int32)  # Convert to int32 for drawing
cv2.polylines(contour_img, [smoothed_contour], isClosed=True, color=255, thickness=2)

# Display the result
plt.imshow(contour_img, cmap='gray')
plt.title('Smoothed Contour')
plt.show()
