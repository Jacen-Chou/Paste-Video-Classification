# coding=utf-8
# import cv2
# import numpy as np
#
# img = cv2.imread("./images_ash_sand_1_16_gamma_test/250/250_1.4.jpg", 0)
#
# # img = cv2.GaussianBlur(img, (3, 3), 0)
# canny = cv2.Canny(img, 10, 30)
#
# cv2.imshow('Canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot

# 加载图片
img = cv2.imread("./images_ash_sand_1_16_gamma_test/250/250_1.4.jpg")

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 移除noisy
img1 = cv2.GaussianBlur(gray, (3, 3), 0)

# 边缘检测
laplacian = cv2.Laplacian(img1, cv2.CV_64F)
sobelx = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)  # x
sobely = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=7)  # y

pyplot.subplot(2, 2, 1), pyplot.imshow(img1, cmap='gray')
pyplot.title('Origin'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(2, 2, 2), pyplot.imshow(laplacian, cmap='gray')
pyplot.title('Laplacian'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(2, 2, 3), pyplot.imshow(sobelx, cmap='gray')
pyplot.title('Sobel X'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(2, 2, 4), pyplot.imshow(sobely, cmap='gray')
pyplot.title('Sobel Y'), pyplot.xticks([]), pyplot.yticks([])

pyplot.show()
