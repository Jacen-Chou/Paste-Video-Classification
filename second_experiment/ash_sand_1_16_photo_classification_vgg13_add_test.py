
# coding: utf-8

# In[ ]:


# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

# 增加了test数据集

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 参数
learning_rate = 0.001
momentum = 0.9
epochs = 50
batch_size = 45
display_step = 1
shuffle = True
num_classes = 45


# In[2]:


# 加载vgg13预训练模型
model = models.vgg13(pretrained=False)
model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                 nn.ReLU(True),
                                 nn.Dropout(),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(True),
                                 nn.Dropout(),
                                 nn.Linear(4096, num_classes))


# In[3]:


# 数据准备
# crop:裁剪 resize:缩放 flip:翻转
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# your image data file
data_dir = './images_paste/images_ash_sand_1_16/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'validation', 'test']}
# torchvision.datasets.ImageFolder返回的是list，这里用torch.utils.data.DataLoader类将list类型的输入数据封装成Tensor数据格式
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size = batch_size,
                                             shuffle = rue,
                                             num_workers = 50) for x in ['train', 'validation']}

dataloders_test = torch.utils.data.DataLoader(image_datasets['test'],
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = 50)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}


# In[4]:


# 是否使用GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

print("use_gpu: " + str(use_gpu))
    
# 定义损失函数，这里采用交叉熵函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化函数，这里采用随机梯度下降法
optimizer = optim.SGD(model.parameters(), learning_rate, momentum)

# 定义学习率的变化策略，这里采用torch.optim.lr_scheduler模块的StepLR类，表示每隔step_size个epoch就将学习率降为原来的gamma倍
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[ ]:


# 开始训练
since = time.time()
best_model_wts = model.state_dict()
best_acc = 0
lowest_loss = 99999
loss_train = [] # 训练集loss
acc_train = [] # 训练集正确率
loss_val = [] # 验证集loss
acc_val = [] # 验证集正确率
best_matrix = [[0 for i in range(num_classes)] for i in range(num_classes)]
f = open('./result/ash_sand_1_16_vgg13_add_test_result.txt', 'a')

for epoch in range(epochs):
    if epoch % display_step == 0:
        print('Epoch [{}/{}]:'.format(epoch + 1, epochs))
        f.write('Epoch [{}/{}]:\n'.format(epoch + 1, epochs))

    # 每一轮都跑一遍训练集和验证集
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()  # 把module设成training模式，对Dropout和BatchNorm有影响
        else:
            model.eval()  # 把module设置为评估模式

        batch_num = 1
        running_loss = 0.0
        running_corrects = 0
        matrix = [[0 for i in range(num_classes)] for i in range(num_classes)]

        # Iterate over data.
        for data in dataloders[phase]:
            # get the inputs
            inputs, labels = data

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # 先将网络中的所有梯度置0
            optimizer.zero_grad()

            # 网络的前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_fn(outputs, labels)

            # 得到模型预测该样本属于哪个类别的信息
            _, preds = torch.max(outputs.data, 1)

            # 训练时，应用回传和优化
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # 记录当前batch_size的loss以及数据对应的分类准确数量
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            if phase == 'validation':
                for k in range(0, num_classes):
                    matrix[labels.data.cpu().numpy()[k]][preds.cpu().numpy()[k]] += 1

            print('\t{} {}-{}: Loss: {:.4f} Acc: {:.4f}%'.format(phase, epoch + 1, batch_num, loss.item()/batch_size, 1.0*torch.sum(preds == labels.data).item()/batch_size*100))
            f.write('\t{} {}-{}: Loss: {:.4f} Acc: {:.4f}%\n'.format(phase, epoch + 1, batch_num, loss.item()/batch_size, 1.0*torch.sum(preds == labels.data).item()/batch_size*100))
            batch_num = batch_num + 1

        # 计算并打印这一轮训练的loss和分类准确率
        if phase == 'train':
            epoch_loss_train = running_loss / dataset_sizes['train']
            epoch_acc_train = running_corrects.item() / dataset_sizes['train']
            loss_train.append(epoch_loss_train)
            acc_train.append(epoch_acc_train)            
        else:
            epoch_loss_val = running_loss / dataset_sizes['validation']
            epoch_acc_val = running_corrects.item() / dataset_sizes['validation']
            loss_val.append(epoch_loss_val)
            acc_val.append(epoch_acc_val)

        if epoch % display_step == 0 and phase == 'validation':
            print('\ttrain Loss: {:.4f} Acc: {:.4f}%'.format(epoch_loss_train, epoch_acc_train*100))
            print('\tvalidation Loss: {:.4f} Acc: {:.4f}%'.format(epoch_loss_val, epoch_acc_val*100))
            f.write('\ttrain Loss: {:.4f} Acc: {:.4f}%\n'.format(epoch_loss_train, epoch_acc_train*100))
            f.write('\tvalidation Loss: {:.4f} Acc: {:.4f}%\n'.format(epoch_loss_val, epoch_acc_val*100))

        # 保存验证集loss最低的模型参数
        if phase == 'validation' and epoch_loss_val < lowest_loss:
            lowest_loss = epoch_loss_val
            best_acc = epoch_acc_val
            best_model_wts = model.state_dict()
            print("Network parameter update.")
            f.write("Network parameter update.\n")
            # 保存最优参数
            torch.save(best_model_wts, './parameter/ash_sand_1_16_vgg13_params.pth')
            best_matrix = copy.deepcopy(matrix)
    time_elapsed = time.time() - since
    print('Time passed {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    print('-' * 20)
    f.write('Time passed {:.0f}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    f.write('-' * 20 + "\n")

# 计算训练所耗时间
time_elapsed = time.time() - since
print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
print('Best validation Acc: {:4f}'.format(best_acc))
f.write('Training complete in {:.0f}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
f.write('Best validation Acc: {:4f}\n'.format(best_acc))
f.close()


# In[8]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
f = open('./result/ash_sand_1_16_vgg13_add_test_result.txt', 'a')

print('loss_train: ' + str(loss_train))
print('loss_val: ' + str(loss_val))
print('acc_train: ' + str(acc_train))
print('acc_val: ' + str(acc_val))
f.write('loss_train: ' + str(loss_train) + '\n')
f.write('loss_val: ' + str(loss_val) + '\n')
f.write('acc_train: ' + str(acc_train) + '\n')
f.write('acc_val: ' + str(acc_val) + '\n')
f.close()

# 绘制第一个图，在一幅图上画两条曲线
plt.figure()
plt.title("Loss",fontsize=16)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(1, 151, 10.0))
plt.plot(range(1,epochs + 1), loss_train,color='r', linewidth = 3.0, label='train')
plt.plot(range(1,epochs + 1), loss_val,color='b', linewidth = 3.0, label='validation')
plt.legend()  # 设置图例和其中的文本的显示

# 绘制第二个图，在一幅图上画两条曲线
plt.figure()
plt.title("Predicted accuracy",fontsize=16)
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.xticks(np.arange(1, 151, 10.0))
plt.plot(range(1,epochs + 1), acc_train,color='r', linewidth = 3.0, label='train')
plt.plot(range(1,epochs + 1), acc_val,color='b', linewidth = 3.0, label='validation')
plt.legend()  # 设置图例和其中的文本的显示

plt.show()


# In[9]:


from openpyxl import Workbook # xlsx


def save(data, path):
    # xlsx
    workbook = Workbook()
    booksheet = workbook.active  # 获取当前活跃的sheet,默认是第一个sheet
    h = len(data) # 行数
    l = len(data[0]) #列数
    for i in range(h):
        for j in range(l):
            booksheet.cell(i+1, j+1).value = data[i][j]
    workbook.save(path)

save(best_matrix,'./result/ash_sand_1_16_vgg13_confusion_matrix.xlsx')


# In[ ]:

# 开始测试
since = time.time()
model.eval()
number = 0
f = open('./result/ash_sand_1_16_vgg13_add_test_result.txt', 'a')
print("=====start test=====")
f.write("=====start test=====")

# Iterate over data.
for data in dataloders_test:
    # get the inputs
    inputs, labels = data
    print('labels: ' + labels.data.cpu().numpy())
    f.write('labels: ' + str(labels.data.cpu().numpy()) + '\n')

    # PyTorch更新至0.4.0后，将Variable和Tensor合并
    if use_gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()

    # 先将网络中的所有梯度置0
    optimizer.zero_grad()

    # 网络的前向传播
    outputs = model(inputs)

    # 计算损失
    loss = loss_fn(outputs, labels)

    # 得到模型预测该样本属于哪个类别的信息
    _, preds = torch.max(outputs.data, 1)
    print('preds: ' + preds.data.cpu().numpy())
    f.write('preds: ' + str(preds.data.cpu().numpy()) + '\n')
    for k in range(batch_size):
        number += 1
        if number == 500:
            number = 0

time_elapsed = time.time() - since
print('Test complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
f.close()