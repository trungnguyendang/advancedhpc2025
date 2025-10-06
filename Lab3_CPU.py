import matplotlib.pyplot as plt
import numpy as np
import time

img = plt.imread("Lab3_image.jpg")

arr = np.array(img)
h, w, _ = img.shape
arr_rgb = arr.reshape(h * w, 3)
start = time.time()
gray_loop = np.zeros(arr_rgb.shape[0], dtype=np.float32)

for i in range(arr_rgb.shape[0]):
    r, g, b = arr_rgb[i]
    gray_loop[i] = 0.299 * r + 0.587 * g + 0.114 * b
end = time.time()

print("CPU execute time:", end - start, "seconds")
gray_img = gray_loop.reshape(h, w)
plt.imsave("grayscale.jpg", gray_img, cmap="gray")

imgplot = plt.imshow(gray_img,cmap="gray")
plt.axis("off")
plt.title('Gray scale image(CPU execute)',fontweight ="bold")
plt.show()

