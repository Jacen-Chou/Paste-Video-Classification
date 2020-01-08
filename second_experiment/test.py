# 用opencv从视频中每隔固定帧数取帧图像
# 训练集1200张，验证集400张，测试集400张

import shutil
import os
import cv2
import numpy as np
import random
import time

img1 = cv2.imread("C:/Users/jacen/Desktop/train_630_3.jpg", 1)
b = np.mean(img1[:,:,0])
g = np.mean(img1[:,:,1])
r = np.mean(img1[:,:,2])

img = cv2.imread("C:/Users/jacen/Desktop/train_630_3.jpg", 1)
b2 = np.mean(img[:,:,0])
g2 = np.mean(img[:,:,1])
r2 = np.mean(img[:,:,2])

print(r,g,b)
print(r2,g2,b2)

rgb_value = img[223, 0]
min_number = min(rgb_value)
print(rgb_value)
print(min_number)
print(img.shape[1] / 2)
print(img.shape[0])

min_number = []
min_number.append(1)
print(min_number)
