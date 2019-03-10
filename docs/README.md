1. 稠密光流的学习
2. 均匀度判断
3. 将现有48类图像做下densenet

#### 20190125 寒假前讨论记录

纯尾砂膏体的图像分类做出来分类正确率是97%

正确率这么高，怀疑是因为各个浓度的样本集各自具有高度的一致性，也就是
每个样本集内部的样本很相似，不同样本集容易区分。所以需要给样本集内部
的样本增加随机量

下一步思路：
+ 训练集和验证集增加随机量（调整亮度、对比度，加噪声，缩放，切割），注意
验证集要事先存好，不能random，避免每次验证的图像不一样，带来误差
+ 验证模型迁移性（用一种膏体的模型去跑另外一种膏体）
+ 为满足工业要求：生产稳定性、速度（每个视频采集固定2min，每秒25帧，
全部送入网络，看得到的预测浓度是不是稳定(stable)，分析两点:1.波动性;2.方差）


#### 20190225 跟师兄讨论

##### 目前进展

1. 发现模型迁移性不好
2. 发现生产稳定性还行，但不是很稳定：http://202.204.62.145:8888/notebooks/ZhouJiacheng/paste_video_classification/second_experiment/images_stability_test.ipynb

##### 师兄指导

1. 做验证生产稳定性实验时候，测试集错误，和训练集有重合，重新再做一次实验

    + 每个视频5min，按照下面要求分割：

        ```
        2min - train 训练集
        1min - validation 验证集
        2min - test 测试集
        6:2:2
        1500
        500
        500
        ```
        
        用验证集求Loss，把验证集中最低的Loss的模型参数保存下来
        
        K-fold K折交叉验证
        
        求方差，浓度按去掉百分号的算，如75%则在计算方差时，用75
        
        尤其是在75%-80%浓度求方差
    + ![](https://i.loli.net/2019/02/25/5c73b48f6c583.jpg)
    
        优化思路：取5个细节，和一个整体缩放，把总共6个112*112*512堆起来，再卷积和分类
        
        进阶：fine-grained


2. 论文及顶会讲解

    + 论文查找：谷歌学术 https://scholar.google.com/
      选择2015年以来的
    + 顶会：https://www.ccf.org.cn/xspj/rgzn/
    + 要发顶会的话，则必须要在公有数据集上验证
    + 尽量别发EI文章

#### 20190305 跟师兄讨论
1. 计算测试集正确率
2. 查找除了方差以外的其他数据稳定性方法，如最大最小差值，差分绝对值之和等
   “中午去图书馆找了找应用统计学里关于离散程度的描述，除了标准差和方差，还有
   很多，如极差range，平均离差 mean deviation，四分位数极差（interquantile range），
   箱线图，偏度，峰度，离散系数（标准差除以平均值），标准分数”
3. 近期去设计如何解决高分辨率图像分类的实验方案

#### 20190307 跟师兄讨论
1. 用新的256*256测试集跑vgg13网络
2. 在1920*1080测试集上滑动窗口224*224，送入网络检测
3. 调整学习率（1）VGG，step,（2）VGG cosin,论文:《Bag of Tricks for Image Classification with Convolutional Neural Networks》
4. 加入SPPNet,论文:《Spatial Pyramid Pooling in Deep Convolutional Networks for visual Recognition》
5. 验证集正确率大于训练集正确率，解决此问题。尝试：把训练集和验证集的预处理方式设置成一样
6. 建议用一张卡跑，可以同时进行多个实验
7. 修改transforms.Normalize参数，mean（平均值）和std（标准差）都自己算，根据1920*1080的原图来算
    以RGB中的R为例，把所有R通道的值加起来，再求'mean=sum÷1920÷1080÷图片数量', 'std=标准差'
    https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=normalize#torchvision.transforms.Normalize
    参考博客代码：https://blog.csdn.net/weixin_38533896/article/details/85951903
8. 增加数据集数量，要求原图1920*1080，训练集3000，验证集1000，测试集1000，按视频时间依次顺序取，最开始一段作训练集，
    中间一段作验证集，最后一段作测试集，不再随机，一共3min20s
9. 更改学习率设置，初步打算每10epochs降低，除以10
10. 除了vgg13，再跑ResNet、DenseNet，作比较，取效果最好的网络
