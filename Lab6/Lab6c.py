import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math

img1 = plt.imread("Image1.jpg").astype(np.float32)
img2 = plt.imread("Image2.jpg").astype(np.float32)
arr1 = np.array(img1)
arr2 = np.array(img2)
# Normalize
img1 /= 255.0
img2 /= 255.0

if img1.shape != img2.shape:
    raise ValueError("Images must have the same dimensions for blending.")

arr1 = img1.reshape(-1)
arr2 = img2.reshape(-1)

dst = np.zeros_like(arr1)
c = 0.7
@cuda.jit
def blend_images_kernel(img1, img2, dst, c):
    tid = cuda.grid(1)
    if tid < img1.size:
        dst[tid] = c * img1[tid] + (1.0 - c) * img2[tid]

threads_per_block = 256
blocks_per_grid = math.ceil(arr1.size / threads_per_block)

d_img1 = cuda.to_device(arr1)
d_img2 = cuda.to_device(arr2)
d_dst = cuda.device_array_like(arr1)

blend_images_kernel[blocks_per_grid, threads_per_block](d_img1, d_img2, d_dst, c)
result = d_dst.copy_to_host()

result = result.reshape(img1.shape)
result = np.clip(result, 0, 1)  # tránh overflow nếu ảnh float32 [0,1]

plt.imshow(result)
plt.title(f"Blended Image (c={c})")
plt.axis('off')
plt.imsave("blended_output.png", result)
plt.show()
