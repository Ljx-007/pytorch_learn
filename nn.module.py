import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("CIFAR10_dataset", False, transform=torchvision.transforms.ToTensor())
test_load = DataLoader(test_data, 64, drop_last=False)


# nn.Module是pytoch提供的神经网络架构，我们要构建自己的神经网络，只需写它的init和forward前向传播即可
class Module(nn.Module):  # 这里子类module继承父类nn.Module
    def __init__(self):
        super().__init__()  # 这里使用super继承了父类的__init__函数
        # 定义卷积层1
        self.conv1 = Conv2d(3, 6, 3, 1, 1)  # 如果不想卷积后图像变小，则需要计算padding的合适值
        # 定义池化层
        self.maxpool1 = MaxPool2d(2, 2, 0)
        # 定义Activation层
        self.relu = ReLU(inplace=False)  # inplace为是否原地替换，若为True，则input经过ReLU后还是返回给input，反之则需要另一个变量来接收返回值，input不变
        self.sigmoid = Sigmoid()

    def forward(self, input):  # 前向传播对input进行卷积操作
        # output = self.conv1(input)
        # output=self.maxpool1(input)
        output = self.sigmoid(input)
        return output


write = SummaryWriter("conv_maxpool_conv")
# 创建Module实例
module = Module()
# 打印module可以看到建立的模型的layer和参数
print(module)
step = 0
for data in test_load:
    imgs, targets = data
    result = module(imgs)
    # print(imgs.shape)
    # print(result.shape)
    # add_images输入的图片需要是3通道的，result是6通道，我们不恰当的用reshape改成三通道的，只是为了验证卷积的结果
    # result = torch.reshape(result, (-1, 3, 30, 30))  # 当通道改为3后，第一维度不知如何变化可以写成-1，电脑会自己算
    write.add_images("sigmoid", result, step)
    step += 1
write.close()
