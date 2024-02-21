import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 这个main是代码执行的入口，与c语言中的main()：类似，当该文件作为主文件执行时，会执行main中的内容，被import则不会执行
# 所以main中可以用来放测试代码，测试神经网络是否正常运行
if __name__ == '__main__':
    model = Model()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)
