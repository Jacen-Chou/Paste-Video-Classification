# -*- coding: utf-8 -*
# 用opencv从视频中每隔固定帧数取帧图像
# 训练集1500张，验证集500张，测试集500张

import shutil
import os
import cv2
import numpy as np
import random
import time

# 视频名字，以浓度命名
density = 200
while True:
    video_path = "./videos_paste/videos_ash_sand_1_16/" + str(density) + ".MOV"
    if not os.path.exists(video_path):
        if density > 780:
            break
        if density < 600:
            density += 50
        else:
            density += 5
        print(str(video_path) + "不存在.")
        continue

    # 如果之前有图片遗留，则清空
    if os.path.exists("./images_paste/images_ash_sand_1_16/train/" + str(density)):
        shutil.rmtree("./images_paste/images_ash_sand_1_16/train/" + str(density))
    if os.path.exists("./images_paste/images_ash_sand_1_16/validation/" + str(density)):
        shutil.rmtree("./images_paste/images_ash_sand_1_16/validation/" + str(density))
    if os.path.exists("./images_paste/images_ash_sand_1_16/test/" + str(density)):
        shutil.rmtree("./images_paste/images_ash_sand_1_16/test/" + str(density))
    time.sleep(1)
    os.mkdir("./images_paste/images_ash_sand_1_16/train/" + str(density))
    os.mkdir("./images_paste/images_ash_sand_1_16/validation/" + str(density))
    os.mkdir("./images_paste/images_ash_sand_1_16/test/" + str(density))

    vc = cv2.VideoCapture(video_path) # 读入视频文件
    c = 1
    num_train = 1
    num_val = 1
    num_test = 1

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    while rval:  # 循环读取视频帧
        if c <= 15 * 25:
            # print(c)
            c = c + 1
            continue

        if num_train > 1500 and num_val > 500:
            break

        timeF = random.randint(1, 2)  # 视频帧计数间隔频率
        while timeF > 0:
            rval, frame = vc.read()
            timeF -= 1

        n = np.random.random()
        if n <= 0.75:
            if num_train <= 1500:
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./images_paste/images_ash_sand_1_16/train/" + str(density) + "/train_" + str(density) +
                            '_' + str(num_train) + '.jpg', frame)  # 存储为图像，训练集，概率0.75
                print(str(density) + ": num_train: %d" % num_train)
                num_train += 1
            else:
                print(str(density) + ": num_train is enough")
                continue
        else:
            if num_val <= 500:
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./images_paste/images_ash_sand_1_16/validation/" + str(density) + "/val_" + str(density) +
                            '_' + str(num_val) + '.jpg', frame)  # 存储为图像，验证集，概率0.25
                print(str(density) + ": num_val: %d" % num_val)
                num_val += 1
            else:
                print(str(density) + ": num_val is enough")
                continue
        # print(c)
        cv2.waitKey(1)

    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./images_paste/images_ash_sand_1_16/test/" + str(density) + "/test_" + str(density) +
                    '_' + str(num_test) + '.jpg', frame)
        print(str(density) + ": num_test: %d" % num_test)
        if num_test == 500:
            break
        num_test += 1
        # print(c)
        cv2.waitKey(1)

    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5
