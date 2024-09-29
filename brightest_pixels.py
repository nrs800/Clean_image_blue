import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to find and plot the 10 brightest pixels
def plot_brightest_pixels(image_path, num_pixels=10):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flatten the grayscale image to a 1D array
    flat_gray = gray_image.flatten()
    
    # Get the indices of the 10 brightest pixels
    brightest_indices = np.argpartition(flat_gray, -num_pixels)[-num_pixels:]
    
    # Convert 1D indices back to 2D coordinates
    brightest_coords = np.unravel_index(brightest_indices, gray_image.shape)
    
    # Plot the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Brightest Pixels")
    
    # Plot the 10 brightest pixels
    plt.scatter(brightest_coords[1], brightest_coords[0], color='red', s=50, label="Brightest Pixels", marker='x')
    plt.legend()
    
    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the image
    image_path = 'image001.png'

    # Plot the brightest 10 pixels
    plot_brightest_pixels(image_path, num_pixels=2)
# ===========================================================================================================

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
plt.figure(figsize=(8,8))



plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu Thresholding')
plt.axis('off')

plt.show()