# 从所有图像中，复制出训练集和验证集图像

import csv
import cv2


# 读取csv至字典
csvFile = open("voc_val.csv", "r")
reader = csv.reader(csvFile)

# 建立空字典
result = {}
i = 0
for item in reader:
    # 忽略第一行
    if reader.line_num == 1:
        continue
    img = cv2.imread("./JPEGImages/" + item[0])
    cv2.imwrite("./val" + item[0], img)
    i += 1
    print(i)

csvFile.close()
