# 膏体均匀度判断

# encoding:utf-8

import numpy as np
import cv2
import os

cap = cv2.VideoCapture('E:/USTB/研究生/1001/膏体项目/videos/75/75_2.mov')

#获取第一帧
ret, frame1 = cap.read()
# frame1 = cv2.resize(frame1, (256, 256), interpolation=cv2.INTER_CUBIC)
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

num = 0

while(1):
    num += 1
    if num % 25 == 0:
        continue
    else:
        ret, frame2 = cap.read()
        # frame2 = cv2.resize(frame2, (256, 256), interpolation=cv2.INTER_CUBIC)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = next - prvs
        diff = cv2.normalize(diff,None,0,255,cv2.NORM_MINMAX)
        print(diff)
        cv2.imshow('diff', diff)
        cv2.imwrite('./frame_diff/diff_' + str(num) + '.jpg', diff)
        # cv2.imwrite('./dense_optical_flow/dy/dy_' + str(num) + '.jpg', dy)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next

cap.release()
cv2.destroyAllWindows()