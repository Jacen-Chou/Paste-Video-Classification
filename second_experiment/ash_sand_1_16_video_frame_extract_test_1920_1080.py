# -_- coding: utf-8 -_
# 用opencv从视频中每隔固定帧数取帧图像
# 测试集1920_1080 从3min开始截取，连续500张

import shutil
import os
import cv2
import numpy as np
import random
import time

# 视频名字，以浓度命名
density = 685
while True:
    video_path = "./videos_paste/videos_ash_sand_1_16/" + str(density) + ".MOV"
    if density == 685:
        video_path = "./videos_paste/videos_ash_sand_1_16/" + str(density) + "_2.MOV"
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
    if os.path.exists("./images_paste/images_ash_sand_1_16/test_1920_1080/" + str(density)):
        shutil.rmtree("./images_paste/images_ash_sand_1_16/test_1920_1080/" + str(density))
    if os.path.exists("./images_paste/images_ash_sand_1_16/test/" + str(density)):
        shutil.rmtree("./images_paste/images_ash_sand_1_16/test/" + str(density))
    time.sleep(1)
    os.mkdir("./images_paste/images_ash_sand_1_16/test_1920_1080/" + str(density))
    os.mkdir("./images_paste/images_ash_sand_1_16/test/" + str(density))

    vc = cv2.VideoCapture(video_path) # 读入视频文件
    c = 1
    num_test = 1

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    while rval:  # 循环读取视频帧

        rval, frame = vc.read()

        # 从3min开始截取帧图像
        if c <= 3 * 60 * 25 and density != 685:
            print("c:" + str(c))
            c = c + 1
            continue

        # 保存原图
        cv2.imwrite("./images_paste/images_ash_sand_1_16/test_1920_1080/" + str(density) + "/test_1920_1080_" + str(density) +
                    '_' + str(num_test) + '.jpg', frame)
        # 尺寸变为256*256，保存
        frame_256 = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./images_paste/images_ash_sand_1_16/test/" + str(density) + "/test_" + str(density) +
                    '_' + str(num_test) + '.jpg', frame_256)
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
