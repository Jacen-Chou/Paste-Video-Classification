# 用opencv从视频中每隔固定帧数取帧图像
# 训练集245张，验证集105张

import shutil
import os
import cv2
import numpy as np
import random
import time

# 视频名字，以浓度命名
density = 200

while True:
    video_path = "F:/paste_videos/灰沙比1比16视频/" + str(density) + ".MOV"
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
    if os.path.exists(video_path):
        shutil.rmtree("F:/paste_videos/images_ash_sand_1_16/train/" + str(density))
        shutil.rmtree("F:/paste_videos/images_ash_sand_1_16/validation/" + str(density))
        shutil.rmtree("F:/paste_videos/images_ash_sand_1_16/test/" + str(density))
    time.sleep(1)
    os.mkdir("F:/paste_videos/images_ash_sand_1_16/train/" + str(density))
    os.mkdir("F:/paste_videos/images_ash_sand_1_16/validation/" + str(density))
    os.mkdir("F:/paste_videos/images_ash_sand_1_16/test/" + str(density))

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

        if num_train > 3000 and num_val > 1000:
            break

        # timeF = random.randint(10, 15)  # 视频帧计数间隔频率
        # while timeF > 0:
        #     rval, frame = vc.read()
        #     timeF -= 1

        rval, frame = vc.read()
        n = np.random.random()
        if n <= 0.7:
            if num_train <= 3000:
                cv2.imwrite("F:/paste_videos/images_ash_sand_1_16/train/" + str(density) + "/train_" + str(density) +
                            '_' + str(num_train) + '.jpg', frame)  # 存储为图像，训练集，概率0.7
                print(str(density) + ": num_train: %d" % num_train)
                num_train += 1
            else:
                print(str(density) + ": num_train is enough")
                continue
        else:
            if num_val <= 1000:
                cv2.imwrite("F:/paste_videos/images_ash_sand_1_16/validation/" + str(density) + "/val_" + str(density) +
                            '_' + str(num_val) + '.jpg', frame)  # 存储为图像，验证集，概率0.3
                print(str(density) + ": num_val: %d" % num_val)
                num_val += 1
            else:
                print(str(density) + ": num_val is enough")
                continue
        # print(c)
        cv2.waitKey(1)

    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5
