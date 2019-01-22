# 用opencv从视频中每隔固定帧数取帧图像

import cv2
import numpy as np

vc = cv2.VideoCapture('E:/USTB/研究生/膏体项目/videos/81/81_1.mov')  # 读入视频文件
c = 1
num_train = 194
num_val = 106

if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False

timeF = 11  # 视频帧计数间隔频率

while rval:  # 循环读取视频帧

    rval, frame = vc.read()

    if c <= 13*25:
        #print(c)
        c = c + 1
        continue

    if c >= 62*25:
        #print(c)
        break

    # cv2.namedWindow("Image")
    # cv2.imshow("Image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if c % timeF == 0:  # 每隔timeF帧进行存储操作
        n = np.random.random()
        if n <= 0.7:
            cv2.imwrite('images/train/81/' + 'train_81_' + str(num_train) + '.jpg', frame)  # 存储为图像，训练集，概率0.7
            print("num_train: %d" % num_train)
            num_train += 1
        else:
            cv2.imwrite('images/validation/81/' + 'val_81_' + str(num_val) + '.jpg', frame)  # 存储为图像，验证集，概率0.3
            print("num_val: %d" % num_val)
            num_val += 1
    #print(c)
    c = c + 1
    cv2.waitKey(1)
