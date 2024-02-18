from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
# 为数据集准备transform操作
dataset_transform=transforms.Compose([
    # 可以有多种操作
    #transforms.Resize...
    #transforms.Normalize...
    transforms.ToTensor()
])

# CIFAR10是pytorch提供的一个物体识别的数据集，参数需要提供一个路径root，train如果为True的话则是将其作为训练集，反之为测试集。
# download=True则代表从网上下载该数据集，如果提供的路径不存在则创建该路径把下载的数据集放在其中，反之代表不需要从网上下载
# 数据集还有一个参数叫transform，代表要对数据集做什么transform的操作
#增加了transform把数据集中的图片全部转为tensor类型
train_set=datasets.CIFAR10("./CIFAR10_dataset",True,transform=dataset_transform,download=True)
test_set=datasets.CIFAR10("./CIFAR10_dataset",False,transform=dataset_transform,download=True)
#打印测试集的第一个元素，包含图片和label，label=3，说明测试集第一张图是第四类——cat
print(test_set[0])
#打印测试集的分类，有10种
print(test_set.classes)
print(len(test_set))
# 再用tensorboard来查看一下数据集
write=SummaryWriter("CIFAR_img")
for i in range(10):
    write.add_image("CIFAR_img",test_set[i][0],i)
write.close()