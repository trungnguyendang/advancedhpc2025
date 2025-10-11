import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math

@cuda.jit
def change_brightness_control(src, dst, value):
    tid = cuda.grid(1)
    if tid < src.size:
        new_val = src[tid] + value
        if new_val > 255:
            new_val = 255
        elif new_val < 0:
            new_val = 0
        dst[tid] = new_val

img = plt.imread("../Image.jpg")

flat = img.flatten()
dst = np.zeros_like(flat)

d_src = cuda.to_device(flat)
d_dst = cuda.device_array_like(flat)

threads_per_block = 256
blocks_per_grid = math.ceil(flat.size / threads_per_block)

# Increase value for brightness, minus for decrease brightness
change_brightness_control[blocks_per_grid, threads_per_block](d_src, d_dst, -50)
cuda.synchronize()


result = d_dst.copy_to_host()
result_reshape = result.reshape(img.shape)
plt.imshow(result_reshape)
plt.axis("off")
plt.title("Gray Image (CUDA GPU Execution)", fontweight="bold")
plt.imsave("GPU_grayscale.jpg", result_reshape, cmap="gray")
plt.show()

