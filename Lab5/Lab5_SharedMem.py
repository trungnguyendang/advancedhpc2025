import numpy as np
import matplotlib.pyplot as plt
import math, time
from numba import cuda, float32
from Lab4.Lab4_GPU import grayscale_kernel
from Lab5 import convolution_with_1D_grid, create_gaussian_kernel

@cuda.jit
def convolution_shared(src, dst, kernel, k_size):
    shared_tile = cuda.shared.array(shape=(0), dtype=float32)
    shared_kernel = cuda.shared.array(shape=(0), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = bx * bw + tx
    y = by * bh + ty

    k_half = k_size // 2
    tile_w = bw + 2 * k_half
    tile_h = bh + 2 * k_half

    shared_tile = cuda.shared.array(shape=(64, 64), dtype=float32)
    shared_kernel = cuda.shared.array(shape=(31, 31), dtype=float32)
    cuda.syncthreads()

    for dy in range(ty, tile_h, bh):
        for dx in range(tx, tile_w, bw):
            gx = bx * bw + dx - k_half
            gy = by * bh + dy - k_half
            if (0 <= gx < src.shape[0]) and (0 <= gy < src.shape[1]):
                shared_tile[dy, dx] = src[gx, gy]
            else:
                shared_tile[dy, dx] = 0.0
    cuda.syncthreads()
    if x < src.shape[0] and y < src.shape[1]:
        value = 0.0
        for i in range(k_size):
            for j in range(k_size):
                value += shared_tile[ty + i, tx + j] * shared_kernel[i, j]
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
    convolution_shared[blocks_per_grid, threads_per_block](d_gray, d_blur, d_kernel, k_size)
    cuda.synchronize()
    end = time.time()

    elapsed = end - start
    times_2d.append(elapsed)
    print(f"2D Block {bs}x{bs}: {elapsed:.4f} s")

threads_per_block_1d = 256
blocks_per_grid_1d = math.ceil((h * w) / threads_per_block_1d)

start = time.time()
grayscale_kernel[blocks_per_grid, (threads_per_block_1d,)](d_img, d_gray, h, w)
convolution_with_1D_grid[blocks_per_grid_1d, threads_per_block_1d](d_gray, d_blur, d_kernel, k_size)
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
