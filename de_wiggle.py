import numpy as np
import cv2
import os

# Function to add noise to images
def add_noise(image):
    noise = np.random.normal(loc=0, scale=25, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Load your images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Example usage
clean_images = load_images_from_folder('/home/nathanael-seay/Documents/VScode_git/clean')
noisy_images = load_images_from_folder('/home/nathanael-seay/Documents/VScode_git/noise')

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the autoencoder model
def build_autoencoder():
    input_img = layers.Input(shape=(None, None, 1))  # Adjust the input shape

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='MAE')
    
    return autoencoder

autoencoder = build_autoencoder()

# Normalize and reshape data
clean_images = np.array(clean_images).astype('float32') / 255.0
noisy_images = np.array(noisy_images).astype('float32') / 255.0

# Reshape for the model (if necessary)
clean_images = clean_images.reshape((clean_images.shape[0], clean_images.shape[1], clean_images.shape[2], 1))
noisy_images = noisy_images.reshape((noisy_images.shape[0], noisy_images.shape[1], noisy_images.shape[2], 1))

# Train the autoencoder
autoencoder.fit(noisy_images, clean_images, epochs=50, batch_size=16, shuffle=False, validation_split=0.1)

# Denoising a test image
test_image = noisy_images[0:1]  # Take a sample noisy image
denoised_image = autoencoder.predict(test_image)

# Display the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(test_image[0].squeeze(), cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(denoised_image[0].squeeze(), cmap='gray')
plt.title('Denoised Image (Autoencoder)')
plt.axis('off')

plt.show()
