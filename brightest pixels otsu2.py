import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the sine function with fixed amplitude and frequency (derived from the number of peaks)
def sine_func(y, phase_shift, offset, amplitude, wavelength):
    return amplitude * np.sin(2 * np.pi * y / wavelength + phase_shift) + offset

# Function to minimize the distance between sine wave and brightest pixels
def objective(params, y_brightest, x_brightest, amplitude, wavelength):
    phase_shift, offset = params
    x_sine = sine_func(y_brightest, phase_shift, offset, amplitude, wavelength)
    return np.sum((x_brightest - x_sine) ** 2)

# Function to find and plot the brightest pixels with a fitted vertical sinusoidal curve on Otsu image
def plot_brightest_pixels_with_fitted_sine(image_path, num_pixels=10):
    # Read the image in color
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise before thresholding
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Flatten the grayscale image to a 1D array
    flat_gray = gray_image.flatten()
    # Get the indices of the brightest pixels
    brightest_indices = np.argpartition(flat_gray, -num_pixels)[-num_pixels:]
    # Convert 1D indices back to 2D coordinates
    brightest_coords = np.unravel_index(brightest_indices, gray_image.shape)
    
    # Extract the x and y coordinates of the brightest pixels
    x_brightest = brightest_coords[1]  # X-coordinates of brightest pixels
    y_brightest = brightest_coords[0]  # Y-coordinates of brightest pixels
    
    # Calculate the bounding box around 95% of the white pixels in the Otsu image
    white_pixels = np.column_stack(np.where(otsu_thresh == 255))
    min_row, max_row = np.percentile(white_pixels[:, 0], [2.5, 97.5]).astype(int)
    min_col, max_col = np.percentile(white_pixels[:, 1], [2.5, 97.5]).astype(int)
    
    # Width of the bounding box (for the sinusoidal amplitude)
    box_width = max_col - min_col
    amplitude = box_width / 2  # Amplitude of the sine wave is half the box width
    
    # Calculate the wavelength based on the number of brightest pixels (1 peak per number of 'X's)
    num_peaks = num_pixels
    wavelength = (max_row - min_row) / num_peaks  # Wavelength for 1 peak per 'X'
    
    # Use optimization to find the best phase shift and offset that minimizes the distance to the red 'X's
    initial_guess = [0, np.mean(x_brightest)]
    result = minimize(objective, initial_guess, args=(y_brightest, x_brightest, amplitude, wavelength), method='L-BFGS-B')
    phase_shift, offset = result.x
    
    # Generate y-coordinates for the curve (smooth interpolation over the range of y_brightest)
    y_curve = np.linspace(min_row, max_row, 1000)
    
    # Use the optimized phase shift and offset to generate the x-coordinates for the sine curve
    x_curve = sine_func(y_curve, phase_shift, offset, amplitude, wavelength)

    # Plot the Otsu threshold image with overlays
    plt.figure(figsize=(8, 8))
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("Otsu Thresholding with Overlays")
    
    # Plot the brightest pixels as red 'X's
    plt.scatter(x_brightest, y_brightest, color='red', s=50, label="Brightest Pixels", marker='x')
    
    # Plot the fitted sinusoidal curve (green) with thicker line
    plt.plot(x_curve, y_curve, color='green', linewidth=4, label="Fitted Sinusoidal Curve")  # Line width set to 4

    # Show the legend and plot
    plt.legend()
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the image
    image_path = 'image001.png'

    # Plot the brightest 2 pixels and the Otsu threshold with a fitted vertical sinusoidal curve
    plot_brightest_pixels_with_fitted_sine(image_path, num_pixels=2)
