import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    r, g, b = src[tidx, 0], src[tidx, 1], src[tidx, 2]
    gray = np.uint8(0.299 * r + 0.587 * g + 0.114 * b)
    dst[tidx, 0] = gray
    dst[tidx, 1] = gray
    dst[tidx, 2] = gray

img = plt.imread("Lab3_image.jpg")  # shape (H, W, 3)

arr = np.array(img)
h, w, c = img.shape
arr_rgb = arr.reshape(h * w, 3)

pixels = arr_rgb.astype(np.uint8)
N = pixels.shape[0]
blockSize = 64
gridSize = N // blockSize

start = time.time()

d_pixels = cuda.to_device(pixels)
d_gray = cuda.device_array_like(pixels)


grayscale[gridSize, blockSize](d_pixels, d_gray)
cuda.synchronize()


gray_img = d_gray.copy_to_host().reshape(h, w, 3)
end = time.time()

print("GPU execute time: ", end - start, "seconds")
plt.imshow(gray_img, cmap="gray")
plt.axis("off")
plt.title('Gray scale image(GPU execute)',fontweight ="bold")
plt.show()
plt.imsave("grayscale_cuda.png", gray_img, cmap="gray")
