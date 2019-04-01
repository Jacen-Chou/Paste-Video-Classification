# -_- coding: utf-8 -_
# 用opencv从视频中每隔固定帧数取帧图像
# 测试集数据集：1920*1080的图像
# 训练集3000，验证集1000，测试集1000
# v1.0 按视频时间依次顺序取，最开始一段作训练集，中间一段作验证集，最后一段作测试集，不用随机，一共3min20s
# 20190318 v2.0 按视频时间依次顺序取，最开始一段作训练集，后面随机取验证集和测试集，一共3min20s

import os
import cv2
import numpy as np

# 视频名字，以浓度命名
density = 685
dir_names = ['train/', 'validation/', 'test/', 'train_224/', 'validation_224/', 'test_224/']
dir_names2 = ['validation/', 'test/', 'validation_224/', 'test_224/']


# 删除一个目录下所有文件和子文件夹中的所有文件
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

while True:
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
    # for dir_name in dir_names2:
    #     path = "./images_paste/images_ash_sand_1_16/" + dir_name
    #     if not os.path.exists(path):
    #         os.mkdir(path)
    #     if os.path.exists(path + str(density)):
    #         del_file(path + str(density))  # 删除文件
    #     else:
    #         os.mkdir(path + str(density))  # 创建文件夹

    vc = cv2.VideoCapture(video_path)  # 读入视频文件
    c = 1
    num_train = 1
    num_val = 939
    num_test = 889

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    # 提取训练集数据
    # while rval:  # 循环读取视频帧
    #     rval, frame = vc.read()
    # #     # # 保存原图
    # #     # cv2.imwrite("./images_paste/images_ash_sand_1_16/train/" + str(density) + "/train_" + str(density) +
    # #     #             '_' + str(num_train) + '.jpg', frame)
    # #     # # 尺寸变为224*224，保存
    # #     # frame_224 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
    # #     # cv2.imwrite("./images_paste/images_ash_sand_1_16/train_224/" + str(density) + "/train_224_" + str(density) +
    # #     #             '_' + str(num_train) + '.jpg', frame_224)
    #     print(str(density) + ": num_train: %d" % num_train)
    #     if num_train == 3000:
    #         break
    #     num_train += 1
    #     cv2.waitKey(1)

    # 提取验证集和测试集数据
    while rval:  # 循环读取视频帧

        if num_val > 1000 and num_test > 1000:
            break

        rval, frame = vc.read()
        n = np.random.random()
        if n < 0.5:
            if num_val <= 1000:
                # 保存原图
                cv2.imwrite("./images_paste/images_ash_sand_1_16/validation/" + str(density) + "/val_" + str(density) +
                            '_' + str(num_val) + '.jpg', frame)
                # 尺寸变为224*224，保存
                frame_224 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./images_paste/images_ash_sand_1_16/validation_224/" + str(density) + "/val_224_" + str(density) +
                            '_' + str(num_val) + '.jpg', frame_224)
                print(str(density) + ": num_val: %d" % num_val)
                num_val += 1
            else:
                print(str(density) + ": num_val is enough")
                continue
        else:
            if num_test <= 1000:
                # 保存原图
                cv2.imwrite("./images_paste/images_ash_sand_1_16/test/" + str(density) + "/test_" + str(density) +
                            '_' + str(num_test) + '.jpg', frame)
                # 尺寸变为224*224，保存
                frame_224 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("./images_paste/images_ash_sand_1_16/test_224/" + str(density) + "/test_224_" + str(density) +
                            '_' + str(num_test) + '.jpg', frame_224)
                print(str(density) + ": num_test: %d" % num_test)
                num_test += 1
            else:
                print(str(density) + ": num_test is enough")
                continue
        cv2.waitKey(1)

    if density == 780:
        break
    if density < 600:
        density += 50
    else:
        density += 5

    break
