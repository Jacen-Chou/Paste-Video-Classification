import numpy as np
import cv2

temp = np.zeros([1080, 1920, 3])

img = cv2.imread('./train_200_1.jpg')
img_np = np.array(img)
temp = temp + img_np
print(img_np[..., 0])
print(temp[..., 0])
print(img_np[..., 0].shape)
print(temp[..., 0].shape)

R = np.copy(img_np[..., 0])
print(R)
print(R.shape)

print(img_np[:, :, 0].sum())
print(R.sum())
print(R.mean())
print(R.mean()/255)
print(R.std())
print(R.std()/255)
