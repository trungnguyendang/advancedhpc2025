import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time
from Lab3.Lab3_GPU import grayscale

img = plt.imread("../Image.jpg")
arr = np.array(img)
h,w,c = img.shape

arr_rgb = arr.reshape(h*w, 3)
pixels = arr_rgb.astype(np.uint8)
N = pixels.shape[0]
blockSize = 256
gridSize = N // blockSize
d_pixels = cuda.to_device(pixels)
d_gray = cuda.device_array_like(pixels)
grayscale[gridSize, blockSize](d_pixels, d_gray)
cuda.synchronize()
gray_host = d_gray.copy_to_host()
gray_img = gray_host.reshape(h, w, 3)

#Define a threshold value τ ∈ [0..255]
threshold_value = 128

#Intensity of pixel Φ(x, y)
binary_img = np.where(gray_img >= threshold_value, 255, 0).astype(np.uint8)

plt.imshow(binary_img, cmap='gray')
plt.axis("off")
plt.title("Binary Image (CUDA GPU Execution)", fontweight="bold")
plt.imsave("GPU_binary.jpg", binary_img, cmap="gray")
plt.show()
plt.savefig("binary_image.png", dpi=300, bbox_inches='tight')