import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

datas = torchvision.datasets.CIFAR10("CIFAR10_dataset", False, torchvision.transforms.ToTensor())
data_load = DataLoader(datas, 64, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2, 2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2, 2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(16 * 64, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = Model()
# 创建一个优化器实例Optimizer
optimizer = torch.optim.SGD(model.parameters(), 0.03)  # SGD是随机梯度下降
Loss = nn.CrossEntropyLoss()
# 开始训练
for epoch in range(20):
    final_loss = 0.
    for data in data_load:  # 这个循环只是把数据集学习了一遍，显然远远不足，于是在外层再套一层循环epcoh
        imgs, targets = data
        output = model(imgs)
        # 计算每个batch的loss值 ，根据这个loss来反向传播计算每层的梯度
        loss = Loss(output, targets)
        # 把一轮每批的loss都加在一起，变成一轮的loss
        final_loss += loss
        # 将每批数据产生的梯度清零，以防未清零的梯度影响到参数的更新
        optimizer.zero_grad()
        # 反向传播计算梯度
        back = loss.backward()
        # 优化器根据梯度更新参数
        optimizer.step()
    # 每5轮打印一次loss值
    if epoch % 5 == 0:
        print(f"Epoch:{epoch} , loss={final_loss}")
