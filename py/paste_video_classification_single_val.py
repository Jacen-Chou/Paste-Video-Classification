# 这个代码训练时不跑验证集，在训练完后跑一次验证集

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os

# 参数
learning_rate = 0.001
momentum = 0.9
epochs = 20
batch_size = 4
display_step = 1
shuffle = True
num_classes = 4

# 加载vgg16预训练模型
model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                 nn.ReLU(True),
                                 nn.Dropout(),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(True),
                                 nn.Dropout(),
                                 nn.Linear(4096, num_classes))

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
}

# your image data file
data_dir = 'E:/PythonProjects/PasteVideoClassification/images'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'validation']}
# torchvision.datasets.ImageFolder返回的是list，这里用torch.utils.data.DataLoader类将list类型的输入数据封装成Tensor数据格式
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size,
                                             shuffle) for x in ['train', 'validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

# 是否使用GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

# 定义损失函数，这里采用交叉熵函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化函数，这里采用随机梯度下降法
optimizer = optim.SGD(model.parameters(), learning_rate, momentum)

# 定义学习率的变化策略，这里采用torch.optim.lr_scheduler模块的StepLR类，表示每隔step_size个epoch就将学习率降为原来的gamma倍
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 开始训练
since = time.time()
best_model_wts = model.state_dict()
best_acc = 0.0

for epoch in range(epochs):

    model.train()  # 把module设成training模式，对Dropout和BatchNorm有影响
    running_loss = 0.0
    running_acc = 0

    # Iterate over data.
    for data in dataloders['train']:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # 先将网络中的所有梯度置0
        optimizer.zero_grad()

        # 网络的前向传播
        outputs = model(inputs)

        # 计算损失
        loss = loss_fn(outputs, labels)

        # 得到模型预测该样本属于哪个类别的信息
        _, predict = torch.max(outputs.data, 1)

        # 训练时，应用回传和优化
        loss.backward()
        optimizer.step()

        # 记录当前的loss以及batchSize数据对应的分类准确数量
        running_loss += loss.item()
        running_acc += torch.sum(predict == labels.data)

    # 计算并打印这一轮训练的loss和分类准确率
    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_acc / dataset_sizes['train']

    # deep copy the model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()
    print('Epoch [{}/{}]: Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epochs, epoch_loss, epoch_acc))

# 计算训练所耗时间
time_elapsed = time.time() - since
print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

# 保存最优参数
model.load_state_dict(best_model_wts)

# ------------------------------------------------------------------
# 开始验证
model.eval() # 把module设置为评估模式

test_loss = 0.0
test_acc = 0

# Iterate over data.
for data in dataloders['validation']:
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # 网络的前向传播
    outputs = model(inputs)

    # 计算损失
    loss = loss_fn(outputs, labels)

    # 得到模型预测该样本属于哪个类别的信息
    _, predict = torch.max(outputs.data, 1)

    # 记录当前的loss以及batchSize数据对应的分类准确数量
    test_loss += loss.data[0]
    test_acc += torch.sum(predict == labels.data)

test_loss = test_loss / dataset_sizes['validation']
test_acc = test_acc / dataset_sizes['validation']

print('Test: Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))







