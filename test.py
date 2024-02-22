# 用来测试模型
import torch
import torchvision
from PIL import Image
from train_model import *

dataset = torchvision.datasets.CIFAR10("CIFAR10_dataset", False)
img = Image.open("./dataset/plane.png")
print(img)
# png类型有四个通道，多了一个透明度，我们创建的model只训练三个通道的图片，所以要把png的四个通道转为三个通道
img = img.convert("RGB")
print(img)
# 用transforms的compose先改变大小再转成tensor
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)
print(img)
model = Model()
model.load_state_dict(torch.load("model_para.pth"))
# model要求输入的维度含有batchsize，所以要reshape一下
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()  # 如果模型中出现dropout或batchnorm等，就要加上
with torch.no_grad():
    output = model(img)
print(output)
pred = output.argmax(1)
print(dataset.classes[pred])
