# 从1080*1920的原图中，按对称位置，裁切五块224*224
# ----0-----------------------1----
# ----------------2----------------
# ----3-----------------------4----

import cv2

img = cv2.imread("./train_200_1.jpg")  # 读取原图
print(img.shape)  # print原图尺寸
cv2.imshow('img', img)  # show原图

# 每一块的x,y坐标
x = [312, 1384, 848, 312, 1384]
y = [102, 102, 428, 754, 754]

# 裁切
for i in range(5):
    cv2.imshow('img' + str(i), img[y[i]:y[i] + 224, x[i]:x[i] + 224])
    print((img[y[i]:y[i] + 224, x[i]:x[i] + 224]).shape)
cv2.waitKey(0)
# cv2.imwrite("./data/cut/cv_cut_thor.jpg", cropped)
