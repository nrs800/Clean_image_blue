import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define image dimensions
height, width = 256, 256

# Function to build CNN model
def build_cnn_model():
    input_img = Input(shape=(height, width, 1))  # Input is a grayscale image

    # First convolutional layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    
    # Second convolutional layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Third convolutional layer
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Output layer: single-channel image
    output_img = Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    
    model = Model(input_img, output_img)
    return model

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(height, width), color_mode='grayscale')
    img = img_to_array(img) / 255.0  # Normalize image to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to load all images from a directory and preprocess them
def load_images_from_directory(directory_path):
    image_list = []
    for filename in sorted(os.listdir(directory_path)):  # Sort to ensure consistent matching
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(directory_path, filename)
            img = load_and_preprocess_image(img_path)
            image_list.append(img)
    return np.vstack(image_list)  # Stack into a single numpy array

# Load images from noisy and clean image directories
def load_image_pairs(noisy_dir, clean_dir):
    noisy_images = load_images_from_directory(noisy_dir)
    clean_images = load_images_from_directory(clean_dir)
    return noisy_images, clean_images

# Example directories for noisy and clean images
noisy_dir = '/home/nathanael-seay/Documents/VScode_git/noise'
clean_dir = '/home/nathanael-seay/Documents/VScode_git/clean'

# Load image pairs
noisy_images, clean_images = load_image_pairs(noisy_dir, clean_dir)

# Build and compile the model
model = build_cnn_model()
model.compile(optimizer='adam', loss='mse')

# Train the model on noisy and clean images
model.fit(noisy_images, clean_images, epochs=20, batch_size=8, validation_split=0.1)

# Load and preprocess a noisy example image for testing
example_noisy_image_path = '/home/nathanael-seay/Documents/VScode_git/noise/noise_wave_3_455.png'
example_noisy_image = load_and_preprocess_image(example_noisy_image_path)

# Use the model to predict the smoothed version of the example noisy image
smoothed_image = model.predict(example_noisy_image)

# Display the noisy and smoothed images for comparison
def display_image_comparison(noisy_img, smooth_img):
    plt.figure(figsize=(10, 5))

    # Display noisy image
    plt.subplot(1, 2, 1)
    plt.title("Noisy Image")
    plt.imshow(noisy_img.squeeze(), cmap='gray')
    plt.axis('off')

    # Display smoothed image
    plt.subplot(1, 2, 2)
    plt.title("Smoothed Image")
    plt.imshow(smooth_img.squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()

# Display the noisy vs. smoothed comparison
display_image_comparison(example_noisy_image, smoothed_image)
