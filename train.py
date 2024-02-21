# 进行一次完整的训练
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_model import *

# 首先加载数据集
train_data = torchvision.datasets.CIFAR10("CIFAR10_dataset", True, torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("CIFAR10_dataset", False, torchvision.transforms.ToTensor())
print(len(train_data), len(test_data))
train_load = DataLoader(train_data, 64, drop_last=True)
test_load = DataLoader(test_data, 64)

# 搭建神经网络
# 将搭建的神经网络放入另一个文件train_model.py中，再import这个文件，能减少代码冗长

# 创建网络模型
model = Model()
# 定义损失函数
Loss = nn.CrossEntropyLoss()
# 定义优化算法
optim = torch.optim.SGD(model.parameters(), lr=0.01)
# 定义迭代次数
num_iteration = 5
# 用tensorboard记录训练过程
write = SummaryWriter("train")
# 定义训练损失和准确率
train_loss = 0
train_accuracy = 0
for epoch in range(num_iteration):
    # 将模型model转换成训练模式，对模型中存在Dropout，BatchNorm等layer起作用
    model.train()
    # 从数据集中依次获取batch
    train_Accuracy = 0
    for data in train_load:
        # 从一个batch中获取图片和label
        imgs, target = data
        # 前向传播
        output = model(imgs)
        # 获取损失值
        train_loss = Loss(output, target)
        # 计算准确率,用argmax获取outputs每行最大值的索引，即获取最有可能的类别，与target做相等运算，返回的是False或True的bool值，把这些bool相加后除以数据集长度，可得准确率
        train_accuracy = ((output.argmax(axis=1) == target).sum())
        # 用train_Accuracy来代表总的准确率，train_accuracy只代表每一个样本的准确率，测试集同理
        train_Accuracy += train_accuracy
        # 梯度更新为0，以免影响到下一个batch的梯度计算
        optim.zero_grad()
        # 反向传播
        train_loss.backward()
        # 优化算法优化模型，更新梯度
        optim.step()

    # 如果想在边训练的时候边测试，可以使用torch.no_grads()，即没有梯度，相当于前向传播
    # 将模型转换成测试模式，对模型中有Dropout，BatchNorm等layer起作用
    model.eval()
    with torch.no_grad():
        test_Accuracy = 0
        for data in test_load:
            imgs, targets = data
            output = model(imgs)
            test_loss = Loss(output, targets)
            # 计算测试集的准确率
            test_accuracy = ((output.argmax(1) == targets).sum())
            test_Accuracy += test_accuracy
    # 每5轮打印一次loss
    if epoch % 5 == 0:
        print(f"Epoch:{epoch},train_Loss:{train_loss},train_accuracy:{(train_accuracy / 64) * 100}%")
        # 记录训练过程
        write.add_scalar("train_loss", train_loss, epoch)
        write.add_scalar("train_accuracy", train_accuracy, epoch)
        write.add_scalar("test_loss", test_loss, epoch)
        write.add_scalar("test_accuracy", test_accuracy, epoch)
# 打印最终结果的loss和准确率
print(f"Train_Loss:{train_loss}  Accuracy:{train_Accuracy * 100 / len(train_data)}%") #训练集训练了不止一次
print(f"Test_Loss{test_loss}  Accuracy:{test_Accuracy * 100 / len(test_data)}%")
write.close()
