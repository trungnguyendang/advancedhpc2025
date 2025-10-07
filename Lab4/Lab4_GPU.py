import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time
import math
@cuda.jit
def grayscale_kernel(src, dst, h, w):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < w and y < h:
        r = src[y, x, 0]
        g = src[y, x, 1]
        b = src[y, x, 2]
        dst[y, x] = (r + g + b) / 3.0

#Load images from lab3
img = plt.imread("../Image.jpg")
h, w, c = img.shape
gray = np.zeros((h, w), dtype=np.float32)
d_img = cuda.to_device(img)
d_gray = cuda.device_array_like(gray)

block_sizes = range(1, 33)
times = []

for bs in block_sizes:
    threads_per_block = (bs, bs)
    blocks_per_grid_x = math.ceil(w / bs)
    blocks_per_grid_y = math.ceil(h / bs)
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    grayscale_kernel[blocks_per_grid, threads_per_block](d_img, d_gray, h, w)
    cuda.synchronize()

    start = time.time()
    grayscale_kernel[blocks_per_grid, threads_per_block](d_img, d_gray, h, w)
    cuda.synchronize()
    end = time.time()

    elapsed_ms = (end - start) * 1000
    times.append(elapsed_ms)
    print(f"Block size {bs}x{bs} → {elapsed_ms:.3f} ms")

#Show the execution time over block size
plt.figure(figsize=(8,4))
plt.plot(block_sizes, times, marker='o', color='blue')
plt.title("Execution Time vs Block Size (2D CUDA Kernel)", fontsize=12, fontweight="bold")
plt.xlabel("Block dimension (N x N)")
plt.ylabel("Execution time (ms)")
plt.grid(True)
plt.show()

#Show the gray img
gray_host = d_gray.copy_to_host()
plt.imshow(gray_host, cmap='gray')
plt.axis("off")
plt.title("Gray Image (CUDA GPU Execution)", fontweight="bold")
plt.show()
