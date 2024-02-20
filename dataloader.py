import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("CIFAR10_dataset", transform=torchvision.transforms.ToTensor(), download=True,
                                         train=False)
# 加载测试集
# shuffle为是否打乱顺序，droplast是当数据集个数与batchsize不能整除时是否留下最后的余数
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)
img, target = test_data[0]
print(img.shape)
print(target)
# 可以通过查看CIFAR10的文档中的getitem方法，得知获取数据集元素时返回的是img，target
# 而在DataLoader中，batch_size是一次性打包多少个数据，此处size=4，则是把每四个元素都打包，返回打包好的imgs和targets,可以用for循环来验证
write=SummaryWriter("dataloader") #用tensorboard来看看打包好的数据
step=0
for epoch in range(2):   #shuffle=True说明每次抓取的数据都不一样，为了验证这一点，我们抓取两次
    for data in test_loader:
        imgs, targets = data
        write.add_images("Epoch:{}".format(epoch),imgs,step)  #多张图片一次输入就用add_images，一张图片就用add_image
        step+=1
write.close()
