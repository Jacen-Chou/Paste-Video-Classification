import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# input = torch.randn(1, 3, 224, 224)
# m = nn.Conv2d(3, 96, 7, stride=2)
# output = m(input)
# print(output.size())

# class SpatialNet(nn.Module):
#
#     def __init__(self):
#         super(SpatialNet, self).__init__()
#
#         # self.conv1 = nn.Conv2d(3, 96, 7, stride=2)
#         # self.conv1_norm = nn.BatchNorm2d(96)
#         # self.conv1_relu = nn.ReLU(inplace=True)
#         # self.conv1_pool = nn.MaxPool2d(kernel_size=3, stride=2)
#         #
#         # self.conv2 = nn.Conv2d(96, 256, 5, stride=2)
#         # self.conv2_norm = nn.BatchNorm2d(256)
#         # self.conv2_relu = nn.ReLU(inplace=True)
#         # self.conv2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
#         #
#         # self.conv3 = nn.Conv2d(256, 512, 3, stride=1)
#         # self.conv3_relu = nn.ReLU(inplace=True)
#         #
#         # self.conv4 = nn.Conv2d(512, 512, 3, stride=1)
#         # self.conv4_relu = nn.ReLU(inplace=True)
#         #
#         # self.conv5 = nn.Conv2d(512, 512, 3, stride=1)
#         # self.conv5_relu = nn.ReLU(inplace=True)
#         # self.conv5_pool = nn.MaxPool2d(kernel_size=3, stride=2)
#
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 96, 7, stride=2),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(96, 256, 5, stride=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(256, 512, 3, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 2 * 2, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 2048),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(2048, 4),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         x = F.softmax(x, dim=1)
#         return x
#
# model = SpatialNet()
# print(model)
#
# inputs = torch.randn(1, 3, 224, 224)
# # print(inputs)
#
# outputs = model(inputs)
#
# print(outputs)
# print(outputs.size())

# --------------------------------------------------


# vis = np.ones((224, 224, 3), np.uint8)*0
# tracks = [[(106.851685, 78.20895), (103.85968, 78.421135), (99.90722, 77.79198), (96.11993, 77.42381), (94.20432, 77.19407), (92.09016, 75.00015), (88.61823, 70.33156), (83.05823, 66.14177), (76.84265, 62.734047), (72.263214, 54.890575)]]
# cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
# cv2.namedWindow("lk_track", 0)
# cv2.resizeWindow("lk_track", 224, 224)
# cv2.imshow('lk_track', vis)
#
# # 按ESC键结束程序
# ch = 0xFF & cv2.waitKey(1)

# ----------------------------------------------------------

a = ([0.2421, 0.2519, 0.2439, 0.2620],
        [0.2506, 0.2578, 0.2408, 0.2508],
        [0.2478, 0.2560, 0.2452, 0.2510],
        [0.2459, 0.2539, 0.2455, 0.2547])
a = np.array(a)
a = torch.from_numpy(a)

b = ([0.2007, 0.1944, 0.1913, 0.4136],
        [0.1485, 0.1500, 0.1672, 0.5343],
        [0.0119, 0.0130, 0.0148, 0.9603],
        [0.1369, 0.1249, 0.1188, 0.6194])
b = np.array(b)
b = torch.from_numpy(b)
#
# print(a)
# print(b)
# print((a+b)/2)

# print(a.data)
# print(b.data)
# print(torch.max(a.data, 1))
# print(torch.max(b.data, 1))
# print(((a+b)/2).data)
# print(torch.max(((a+b)/2).data, 1))

# ==========================================================
# for a, b in zip([1,2,3,4,5], [6,7,8,9,10]):
#     print(a)
#     print(b)

# ==========================================================

a = ([3, 3, 3, 3])
a = np.array(a)
a = torch.from_numpy(a)

b = ([3, 1, 1, 3])
b = np.array(b)
b = torch.from_numpy(b)

print(torch.sum(a.data == b.data))