import torch
import torch.nn as nn
import torch.nn.functional as F

# input = torch.randn(1, 3, 224, 224)
# m = nn.Conv2d(3, 96, 7, stride=2)
# output = m(input)
# print(output.size())

class SpatialNet(nn.Module):

    def __init__(self):
        super(SpatialNet, self).__init__()

        # self.conv1 = nn.Conv2d(3, 96, 7, stride=2)
        # self.conv1_norm = nn.BatchNorm2d(96)
        # self.conv1_relu = nn.ReLU(inplace=True)
        # self.conv1_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        #
        # self.conv2 = nn.Conv2d(96, 256, 5, stride=2)
        # self.conv2_norm = nn.BatchNorm2d(256)
        # self.conv2_relu = nn.ReLU(inplace=True)
        # self.conv2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        #
        # self.conv3 = nn.Conv2d(256, 512, 3, stride=1)
        # self.conv3_relu = nn.ReLU(inplace=True)
        #
        # self.conv4 = nn.Conv2d(512, 512, 3, stride=1)
        # self.conv4_relu = nn.ReLU(inplace=True)
        #
        # self.conv5 = nn.Conv2d(512, 512, 3, stride=1)
        # self.conv5_relu = nn.ReLU(inplace=True)
        # self.conv5_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 7, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x

model = SpatialNet()
print(model)

inputs = torch.randn(1, 3, 224, 224)
# print(inputs)

outputs = model(inputs)

print(outputs)
print(outputs.size())