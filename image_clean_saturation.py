import cv2
import numpy as np

# Function to segment based on saturation
def saturation_segmentation(image_path, lower_saturation, upper_saturation):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the saturation channel (second channel in HSV)
    saturation = hsv_image[:, :, 1]
    
    # Define the mask based on the saturation range
    mask = cv2.inRange(saturation, lower_saturation, upper_saturation)
    
    # Segment the image using the mask
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented_image, mask

# Example usage
if __name__ == "__main__":
    # Path to the image
    image_path = 'image001.png'

    # Define lower and upper saturation bounds for segmentation
    # Range for segmentation (adjust these values as needed)
    lower_saturation = 50
    upper_saturation = 254.5

    # Get segmented image
    segmented_image, mask = saturation_segmentation(image_path, lower_saturation, upper_saturation)

    # Show the images
    cv2.imshow('Original Image', cv2.imread(image_path))
    cv2.imshow('Mask', mask)
    cv2.imshow('Segmented Image', segmented_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
