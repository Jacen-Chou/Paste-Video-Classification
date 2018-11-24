# Paste-Video-Classification
膏体视频分类

##### 数据集

读取视频流，提取视频帧（cv2.imwrite）。将4个不同类别的视频每秒材3帧图像（采样间隔待定），按照7:3的比率（numpy随机数判断）分成训练集和验证集（在提取到帧的同时，立马生成随机数，然后存入对应文件夹），并生成对应图像的标签（“CSV文件（ https://pytorch.org/tutorials/beginner/data_loading_tutorial.html ）”，“标记浓度标签（用torchvision.datasets.ImageFolder，分文件夹”），最终得到图片集

##### 模型

对比VGG13、VGG16、VGG19、ResNet50、ResNet101、ResNet152、DenseNet121 在相同参数下的实验情况（ https://pytorch.org/docs/master/torchvision/models.html ）

##### 参数

学习率，动量等设置一致，均在40个Epoch下查看验证集的正确率

##### 实验对比方法 

使用matplolib绘制曲线，对比不同方法的验证集正确率，查看出收敛快慢和实验效果 

##### 数据保存 

1、最好结果的模型参数保存；2、每轮Loss数据保存；

3、每轮正确率保存；4、最高正确率保存