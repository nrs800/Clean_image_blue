import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Function to generate sine waves
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = np.sin(2 * np.pi * freq * x)
    return x, y

# Function to add random high-frequency noise
def add_random_high_frequency_noise(signal, sample_rate, num_noise_components=5, noise_amplitude=0.5):
    """
    Adds random high-frequency noise to the signal.
    
    Parameters:
    - signal: The original sine wave signal.
    - sample_rate: Number of samples per second.
    - num_noise_components: Number of random high-frequency components to add.
    - noise_amplitude: The amplitude of the noise.
    
    Returns:
    - noisy_signal: The signal with added random high-frequency noise.
    """
    x = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    noisy_signal = signal.copy()

    # Add multiple random high-frequency components
    for _ in range(num_noise_components):
        # Randomly choose a high frequency and phase shift
        random_freq = np.random.uniform(20, 50)  # High frequencies between 30 and 100 Hz
        random_phase = np.random.uniform(0, 2 * np.pi)  # Random phase shift
        
        # Generate the high-frequency noise component
        high_freq_noise = noise_amplitude * np.sin(2 * np.pi * random_freq * x + random_phase)
        
        # Add the noise to the signal
        noisy_signal += high_freq_noise
    
    return noisy_signal

# Parameters for the sine wave
frequencies = []  # Frequencies of the sine waves in Hz
count = 500
for j in range(count):
    frequency = random.randint(1, 10)   # Constant frequency for all waves
    frequencies.append(frequency)  # Correct way to append frequencies

sample_rate = 1000  # Samples per second
duration = 2.0  # Duration in seconds

# Parameters for the random high-frequency noise
num_noise_components = 5  # Number of random high-frequency components
noise_amplitude = 0.2  # Amplitude of the random noise

# Directory to store the images
output_dir = "sine_wave_images_with_random_noise2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir_clean = "sine_wave_images_2"
if not os.path.exists(output_dir_clean):
    os.makedirs(output_dir_clean)

# Generate and save the sine waves with random high-frequency noise as images
for i, freq in enumerate(frequencies):
    # Generate sine wave
    x, sine_wave = generate_sine_wave(freq, sample_rate, duration)
    
    # Add random high-frequency noise
    noisy_wave = add_random_high_frequency_noise(sine_wave, sample_rate, num_noise_components, noise_amplitude)
    clean_wave = add_random_high_frequency_noise(sine_wave, sample_rate, 0, 0)
    # Plot the sine wave with random high-frequency noise
    plt.figure(figsize=(10, 6))
    plt.plot(x, noisy_wave, label=f'Sine wave {freq} Hz with random noise')
    # plt.title(f'Sine Wave {freq} Hz with Random High-Frequency Noise')
   # plt.xlabel('Time [s]')
    #plt.ylabel('Amplitude')
    #plt.legend()
    plt.grid(False)
    
    # Save the plot as an image
    image_path = os.path.join(output_dir, f'noise_wave_{freq}_{i}.png')  # Use index to differentiate images
    plt.savefig(image_path)
    plt.close()  # Close the figure after saving to avoid memory issues

    print(f"Image saved: {image_path}")

# -------------------------------------------------------------------------------------------------------------------------------
    # Plot the sine wave with random high-frequency noise
    plt.figure(figsize=(10, 6))
    plt.plot(x, clean_wave, label=f'Sine wave {freq} Hz')
    # plt.title(f'Sine Wave {freq} Hz with Random High-Frequency Noise')
   # plt.xlabel('Time [s]')
    #plt.ylabel('Amplitude')
    #plt.legend()
    plt.grid(False)
    
    # Save the plot as an image
    image_path2 = os.path.join(output_dir_clean, f'sine_wave_{freq}_{i}.png')  # Use index to differentiate images
    plt.savefig(image_path2)
    plt.close()  # Close the figure after saving to avoid memory issues

    print(f"Image saved: {image_path2}")
