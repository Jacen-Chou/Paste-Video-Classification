# encoding:utf-8

# Lucas–Kanade光流算法

# calcOpticalFlowPyrLK方法
# nextPts, status, err = calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, err[, winSize[,
#                                               maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]])
# 参数说明：
# prevImage 前一帧8-bit图像
# nextImage 当前帧8-bit图像
# prevPts 待跟踪的特征点向量
# nextPts 输出跟踪特征点向量
# status 特征点是否找到，找到的状态为1，未找到的状态为0
# err 输出错误向量
# winSize 搜索窗口的大小
# maxLevel 最大的金字塔层数

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

import numpy as np
import cv2
# from common import anorm2, draw_str
from time import clock

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):  # 构造方法，初始化一些参数和视频路径
        self.track_len = 10
        self.detect_interval = 1
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):  # 光流运行方法
        i = 1
        num_train = 224
        num_val = 91
        while True:
            ret, frame = self.cam.read()  # 读取视频帧
            if ret == True: # 判断视频读取是否结束
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                vis = np.ones((256, 256, 3), np.uint8)*0 # 新建黑色画布(0)，改为255则变为白色画布
                # vis = frame.copy() # 以原来帧图像作为背景

                if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                           **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                            **lk_params)  # 当前帧跟踪到的角点和前一帧的图像作为输入来找到前一帧的角点位置
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
                    good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        # 在跟踪点画圆，即标注起点
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks

                    # 以上一帧角点为初始点，当前帧跟踪到的点为终点划线
                    # (0, 255, 0)表示绿色
                    # img:图像,顶点集,是否闭合,颜色
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False,
                                  (0, 255, 0))
                    # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                # 检测角点
                if self.frame_idx % self.detect_interval == 0:  # 每1帧检测一次特征点，即角点
                    mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
                    mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    # 像素级别角点检测，goodFeaturesToTrack这个函数使用的是Shi-Tomasi角点检测算子
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray

                cv2.namedWindow("lk_track", 0)
                cv2.resizeWindow("lk_track", 256, 256)
                cv2.imshow('lk_track', vis)

                # 存储为图像
                i += 1
                
                # 视频去头去尾
                if i < 50 or i > 1175:
                    print("pass.")
                    continue
                
                # cv2.imwrite('./images/' + str(i) + '.jpg', vis)
                # cv2.imwrite('./flow_images/' + str(i) + '.jpg', vis)
                if i % 19 == 0:  # 每隔timeF帧进行存储操作
                    n = np.random.random()
                    if n <= 0.7:
                        cv2.imwrite('../flow_images/train/75/' + 'flow_train_75_' + str(num_train) + '.jpg',
                                    vis)  # 存储为图像，训练集，概率0.7
                        print("num_train: %d" % num_train)
                        num_train += 1
                    else:
                        cv2.imwrite('../flow_images/validation/75/' + 'flow_val_75_' + str(num_val) + '.jpg',
                                    vis)  # 存储为图像，验证集，概率0.3
                        print("num_val: %d" % num_val)
                        num_val += 1

            # 按ESC键结束程序
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = "E:/USTB/研究生/1001/膏体项目/videos/75/75_2.mov"
        # video_src = "./75_1.MOV"

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
