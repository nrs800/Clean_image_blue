from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage import io

# Load a binary image
img = io.imread('/home/nathanael-seay/Documents/Otsu image cropped.png', as_gray=True)

# Thresholding to ensure binary image
binary = img > 0.6

# Perform skeletonization
skeleton = skeletonize(binary)

# Display the skeleton
plt.imshow(skeleton, cmap='gray')
plt.title('Skeletonized Image')
plt.show()
