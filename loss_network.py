import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

datas = torchvision.datasets.CIFAR10("CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(datas, 64, drop_last=True)


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
Loss=nn.CrossEntropyLoss()
for data in data_loader:
    imgs, targets = data
    output = model(imgs)
    #计算每个batch的loss值
    loss=Loss(output,targets)
    #根据这个loss来反向传播计算每层的梯度
    back=loss.backward()
    print(loss)
