# 用opencv从视频中每隔固定帧数取帧图像
# 训练集600张，验证集200张，测试集200张
# 尺寸1920*1080

# img.shape[0]：图像的垂直尺寸（高度）
# img.shape[1]：图像的水平尺寸（宽度）
# img.shape[2]：图像的通道数
# 在矩阵中，[0]就表示行数，[1]则表示列数。

import shutil
import os
import cv2
import numpy as np
import random
import time

# 63 63.5 64 64.5 65 65.5 66 66.5 67
# 视频名字，以浓度命名
video_root_path = "D:/nfca_data/nfca_paste_videos/"
image_root_path = "D:/nfca_data/nfca_paste_images_1920/"
density_dict = {
    630: "DSC_4940",
    635: "DSC_4936",
    640: "DSC_4916",
    645: "DSC_4937",
    650: "DSC_4951",
    655: "DSC_4947",
    660: "DSC_4934",
    665: "DSC_4931",
    670: "DSC_4939"
}
density = 630
while True:
    video_path = video_root_path + density_dict[density] + ".MOV"

    # 如果之前有图片遗留，则清空
    if os.path.exists(image_root_path + "train/" + str(density)):
        shutil.rmtree(image_root_path + "train/" + str(density))
        shutil.rmtree(image_root_path + "validation/" + str(density))
        shutil.rmtree(image_root_path + "test/" + str(density))
    time.sleep(1)
    os.makedirs(image_root_path + "train/" + str(density))
    os.makedirs(image_root_path + "validation/" + str(density))
    os.makedirs(image_root_path + "test/" + str(density))

    # 起始帧
    frame_start = {
        630: 18609,
        635: 28801,
        640: 14400,
        645: 1,
        650: 14400,
        655: 1,
        660: 28801,
        665: 28801,
        670: 1
    }
    # 结束帧
    frame_stop = {
        630: 28800,
        635: 43200,
        640: 28800,
        645: 14400,
        650: 28800,
        655: 14400,
        660: 43200,
        665: 43200,
        670: 12270
    }
    # 当前截取图像的帧
    frame_now = frame_start[density]
    # 帧间隔
    frame_step = 5
    num_train = 1
    num_val = 1
    num_test = 1

    vc = cv2.VideoCapture(video_path)  # 读入视频文件
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_start[density])  # 设置要获取的帧号
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        print("open video error!")
        rval = False

    while rval:  # 循环读取视频帧

        if num_train > 600 and num_val > 200:
            break

        if frame_now > frame_stop[density]:
            print("now at the last frame.")
            break

        n = np.random.random()
        if n <= 0.75:
            if num_train <= 600:
                cv2.imwrite(image_root_path + "train/" + str(density) + "/train_" + str(density) +
                            '_' + str(num_train) + '.jpg', frame)  # 存储为图像，训练集，概率0.75
                print(str(density) + ": num_train: %d" % num_train)
                num_train += 1
                if num_train > 600:
                    continue
            else:
                print(str(density) + ": num_train is enough")
                continue
        else:
            if num_val <= 200:
                cv2.imwrite(image_root_path + "validation/" + str(density) + "/val_" + str(density) +
                            '_' + str(num_val) + '.jpg', frame)  # 存储为图像，验证集，概率0.25
                print(str(density) + ": num_val: %d" % num_val)
                num_val += 1
                if num_val > 200:
                    continue
            else:
                print(str(density) + ": num_val is enough")
                continue

        frame_now = frame_now + frame_step  # 每隔frame_step帧搅拌机位置一样，所以每隔frame_step帧取一次
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_now)  # 设置要获取的帧号
        rval, frame = vc.read()
        cv2.waitKey(1)

    if frame_now > frame_stop[density]:
        print("now at the last frame.")
        break

    while rval:  # 循环读取视频帧
        frame_now = frame_now + frame_step  # 每隔frame_step帧搅拌机位置一样，所以每隔frame_step帧取一次
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_now)  # 设置要获取的帧号
        rval, frame = vc.read()
        cv2.imwrite(image_root_path + "test/" + str(density) + "/test_" + str(density) +
                    '_' + str(num_test) + '.jpg', frame)  # 存储为图像，测试集
        print(str(density) + ": num_test: %d" % num_test)
        num_test += 1
        if num_test > 200:
            break
        cv2.waitKey(1)

    if density == 670:
        break
    else:
        density += 5
