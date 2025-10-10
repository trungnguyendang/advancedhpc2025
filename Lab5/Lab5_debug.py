import numpy as np
import matplotlib.pyplot as plt
import math, time
from numba import cuda, float32
from Lab4.Lab4_GPU import grayscale_kernel  # assume same interface

def create_gaussian_kernel(size, sigma):
    k = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    sum_val = 0.0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            val = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            k[i, j] = val
            sum_val += val
    k /= sum_val
    return k

@cuda.jit
def convolution(src, dst, kernel, k_size):
    x, y = cuda.grid(2)
    h, w = src.shape
    if x < h and y < w:
        k_half = k_size // 2
        value = 0.0
        for i in range(-k_half, k_half + 1):
            for j in range(-k_half, k_half + 1):
                nx = x + i
                ny = y + j
                if (0 <= nx < h) and (0 <= ny < w):
                    value += src[nx, ny] * kernel[i + k_half, j + k_half]
        dst[x, y] = value

@cuda.jit
def convolution_1D(src, dst, kernel, k_size):
    idx = cuda.grid(1)
    h, w = src.shape
    if idx < h * w:
        x = idx // w
        y = idx % w
        k_half = k_size // 2
        value = 0.0
        for i in range(-k_half, k_half + 1):
            for j in range(-k_half, k_half + 1):
                nx = x + i
                ny = y + j
                if (0 <= nx < h) and (0 <= ny < w):
                    value += src[nx, ny] * kernel[i + k_half, j + k_half]
        dst[x, y] = value


img = plt.imread("../Image.jpg").astype(np.float32)
h, w, c = img.shape

k_size = 21
sigma = 5
kernel = create_gaussian_kernel(k_size, sigma)

d_img = cuda.to_device(img)
d_gray = cuda.device_array((h, w), dtype=np.float32)
d_blur = cuda.device_array_like(d_gray)
d_kernel = cuda.to_device(kernel)


block_sizes = [4, 8, 16, 32]
times_2d = []

for bs in block_sizes:
    threads_per_block = (bs, bs)
    blocks_per_grid = (math.ceil(w / bs), math.ceil(h / bs))

    start = time.time()
    grayscale_kernel[blocks_per_grid, threads_per_block](d_img, d_gray, h, w)
    convolution[blocks_per_grid, threads_per_block](d_gray, d_blur, d_kernel, k_size)
    cuda.synchronize()
    end = time.time()

    elapsed = end - start
    times_2d.append(elapsed)
    print(f"2D Block {bs}x{bs}: {elapsed:.4f} s")

threads_per_block_1d = 256
blocks_per_grid_1d = math.ceil((h * w) / threads_per_block_1d)

start = time.time()
grayscale_kernel[blocks_per_grid, (threads_per_block_1d,)](d_img, d_gray, h, w)
convolution_1D[blocks_per_grid_1d, threads_per_block_1d](d_gray, d_blur, d_kernel, k_size)
cuda.synchronize()
end = time.time()

time_1d = end - start
print(f"1D Grid: {time_1d:.4f} s")

speedup = [time_1d / t for t in times_2d]

plt.figure(figsize=(8, 5))
plt.plot(block_sizes, speedup, marker='o')
plt.title("Speedup vs Block Size (2D Grid vs 1D Grid)")
plt.xlabel("Block size")
plt.ylabel("Speedup over 1D grid")
plt.grid(True)
plt.savefig("speed_vs_blockSize.png", dpi=300, bbox_inches='tight')
plt.show()

blurred = d_blur.copy_to_host()
gray = d_gray.copy_to_host()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Gray")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Gaussian Blurred (best 2D config)")
plt.imshow(blurred, cmap='gray')
plt.axis('off')
plt.show()
