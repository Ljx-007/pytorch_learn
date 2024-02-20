import torch
import torch.nn as nn
import torchvision.datasets
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("CIFAR10_dataset", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("CIFAR10_dataset", False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(3, 32, 5, 1, 2)
        # # 如果MaxPool2d只填一个参数，则默认kernelsize和stride相同
        # self.maxpool1 = MaxPool2d(2, 2)
        # self.conv2 = Conv2d(32, 32, 5, 1, 2)
        # self.maxpool2 = MaxPool2d(2, 2)
        # self.conv3 = Conv2d(32, 64, 5, 1, 2)
        # self.maxpool3 = MaxPool2d(2, 2)
        # self.flatten = Flatten()  # 展平，作用与torch.flatten相同
        # self.dense1 = Linear(16 * 64, 64)
        # self.dense2 = Linear(64, 10)
        #当网络层数很多时，写forward函数要重新写一遍，这样代码显得很臃肿，于是使用Sequential函数将以上layer打包在一起，作用与transforms.compose类似
        self.model=Sequential(
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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.dense1(x)
        # x = self.dense2(x)
        #使用Sequential之后以上代码就不用重复写了
        x=self.model(x)
        return x


model = Model()
print(model)
#设置一个例子来验证一下网络是否正确
input=torch.ones((64,3,32,32)) #创建一个全是1的矩阵，维度为(64,3,32,32)
output=model(input)
print(output.shape)
#建好的模型做成图还可以用tensorboard来看
write=SummaryWriter("model")
write.add_graph(model,input) #第一个参数是建好的model实例，第二个是model的输入
write.close()
