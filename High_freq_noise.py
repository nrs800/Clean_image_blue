
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to generate sine waves
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = np.sin(2 * np.pi * freq * x)
    return x, y

# Function to add high-frequency noise
def add_high_frequency_noise(signal, noise_amplitude, noise_freq, sample_rate):
    x = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    high_freq_noise = noise_amplitude * np.sin(2 * np.pi * noise_freq * x)
    noisy_signal = signal + high_freq_noise
    return noisy_signal

# Parameters for the sine wave
frequencies = [2, 2, 2]  # Frequencies of the sine waves in Hz
sample_rate = 1000  # Samples per second
duration = 2.0  # Duration in seconds

# Parameters for the high-frequency noise
noise_amplitude = 0.5  # Amplitude of the noise
noise_freq = 10  # Frequency of the high-frequency noise in Hz

# Directory to store the images
output_dir = "sine_wave_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate and save the sine waves with high-frequency noise as images
for i, freq in enumerate(frequencies):
    # Generate sine wave
    x, sine_wave = generate_sine_wave(freq, sample_rate, duration)
    
    # Add high-frequency noise
    noisy_wave = add_high_frequency_noise(sine_wave, noise_amplitude, noise_freq, sample_rate)
    
    # Plot the sine wave with high-frequency noise
    plt.figure(figsize=(10, 6))
    plt.plot(x, noisy_wave, label=f'Sine wave {freq} Hz with noise')
    plt.title(f'Sine Wave {freq} Hz with High-Frequency Noise')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
    image_path = os.path.join(output_dir, f'sine_wave_{freq}_Hz_with_noise.png')
    plt.savefig(image_path)
    plt.close()  # Close the figure after saving to avoid memory issues

    print(f"Image saved: {image_path}")
