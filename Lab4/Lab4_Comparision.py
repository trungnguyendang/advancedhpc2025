import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time
import math
import Lab3.Lab3_GPU as lab3

@cuda.jit
def grayscale_kernel_1d(src, dst, total_pixels):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if x < total_pixels:
        r = src[x, 0]
        g = src[x, 1]
        b = src[x, 2]
        dst[x] = (r + g + b) / 3.0

@cuda.jit
def grayscale_kernel_2d(src, dst, h, w):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < w and y < h:
        r = src[y, x, 0]
        g = src[y, x, 1]
        b = src[y, x, 2]
        dst[y, x] = (r + g + b) / 3.0

img = plt.imread("../Image.jpg")
h, w, c = img.shape

img_flat = img.reshape(-1, 3)
total_pixels = img_flat.shape[0]

gray_2d = np.zeros((h, w), dtype=np.float32)
gray_1d = np.zeros((total_pixels, 3), dtype=np.float32)

d_img_2d = cuda.to_device(img)
d_gray_2d = cuda.device_array_like(gray_2d)

d_img_1d = cuda.to_device(img_flat)
d_gray_1d = cuda.device_array_like(gray_1d)


block_sizes = range(1, 33)
times_1d = []
times_2d = []

for bs in block_sizes:
    #1D Kernel
    threads_per_block = bs
    blocks_per_grid = total_pixels // threads_per_block

    lab3.grayscale[blocks_per_grid, threads_per_block](d_img_1d, d_gray_1d)
    cuda.synchronize()

    start = time.time()
    lab3.grayscale[blocks_per_grid, threads_per_block](d_img_1d, d_gray_1d)
    cuda.synchronize()
    end = time.time()
    times_1d.append((end - start) * 1000)

    #2D Kernel
    threads_per_block = (bs, bs)
    blocks_per_grid_x = math.ceil(w / bs)
    blocks_per_grid_y = math.ceil(h / bs)
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    grayscale_kernel_2d[blocks_per_grid, threads_per_block](d_img_2d, d_gray_2d, h, w)
    cuda.synchronize()

    start = time.time()
    grayscale_kernel_2d[blocks_per_grid, threads_per_block](d_img_2d, d_gray_2d, h, w)
    cuda.synchronize()
    end = time.time()
    times_2d.append((end - start) * 1000)
    print(f"Block {bs}x{bs}: 1D = {times_1d[-1]:.3f} ms, 2D = {times_2d[-1]:.3f} ms")

plt.figure(figsize=(8, 5))
plt.plot(block_sizes, times_1d, marker='o', label='1D Kernel')
plt.plot(block_sizes, times_2d, marker='s', label='2D Kernel')
plt.title("Execution Time Comparison: 1D vs 2D CUDA Kernel", fontsize=12, fontweight="bold")
plt.xlabel("Block size (N or N×N)")
plt.ylabel("Execution time (ms)")
plt.grid(True)
plt.legend()
plt.savefig("execute_time_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
