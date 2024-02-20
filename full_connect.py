import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader

datas = torchvision.datasets.CIFAR10("CIFAR10_dataset", False, torchvision.transforms.ToTensor())
load = DataLoader(datas, 64, drop_last=True)


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 4096, 32)
        # 定义一个全连接层，在pytorch中全连接层叫Linear
        self.full_conect = Linear(4096 * 64, 10)  # 第一个参数为前一层的特征个数，后一个参数为该层的特征个数

    def forward(self, input):
        output = self.conv(input)
        # 卷积完后output的shape是(64,4096,1,1),在经过全连接层时需要展平，使用flatten函数
        output = torch.flatten(output)
        # 展平后经过全连接层
        output = self.full_conect(output)
        return output


module = Module()
for data in load:
    imgs, targets = data
    output = module(imgs)
    print(imgs.shape)
    print(output.shape)
