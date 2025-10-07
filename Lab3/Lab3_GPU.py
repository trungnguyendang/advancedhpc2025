import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx < src.shape[0]:
        r, g, b = src[tidx, 0], src[tidx, 1], src[tidx, 2]
        gray = (r + g + b) / 3.0
        dst[tidx, 0] = gray
        dst[tidx, 1] = gray
        dst[tidx, 2] = gray

img = plt.imread("../Image.jpg")  # shape (H, W, 3)

arr = np.array(img)
h, w, c = img.shape
arr_rgb = arr.reshape(h * w, 3)

pixels = arr_rgb.astype(np.uint8)
N = pixels.shape[0]

block_sizes = range(1, 1024)
times = []
d_pixels = cuda.to_device(pixels)
d_gray = cuda.device_array_like(pixels)

for blockSize in block_sizes:
    pixelCount = h * w
    gridSize = pixelCount // blockSize

    start = time.time()
    grayscale[gridSize, blockSize](d_pixels, d_gray)
    cuda.synchronize()
    end = time.time()

    elapsed = end - start
    times.append(elapsed)
    print(f"blockSize={blockSize}, GPU time={elapsed:.6f} sec")

plt.plot(block_sizes, times, marker='o', linestyle='-', color='b')
plt.xlabel('Block Size (threads per block)', fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontweight='bold')
plt.title('GPU Grayscale Execution Time vs Block Size', fontweight='bold')
plt.grid(True)
# save the plot as an image file
plt.savefig("blocksize_vs_time.png", dpi=300, bbox_inches='tight')
plt.show()

gray_host = d_gray.copy_to_host()
gray_img = gray_host.reshape(h, w, 3)
plt.imshow(gray_img, cmap='gray')
plt.axis("off")
plt.title("Gray Image (CUDA GPU Execution)", fontweight="bold")
plt.imsave("GPU_grayscale.jpg", gray_img, cmap="gray")
plt.show()
