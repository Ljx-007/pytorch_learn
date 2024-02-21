import torchvision.models
from torch import nn
from torchvision.models import VGG16_Weights

# pytorch有自带的很多模型，这里导入VGG16，如果想要使用预训练好的参数，则查文档可得需要填的参数，否则默认为不使用预训练参数
VGG16 = torchvision.models.vgg16(VGG16_Weights.DEFAULT)
print(VGG16)
# 在VGG16的基础上自己进行增加，修改，删除预定层
# 增加一层
VGG16.add_module('add_layer', nn.Linear(1000, 10))
print(VGG16)
# 在指定位置classifier增加一层
VGG16.classifier.add_module("add_layer", nn.Linear(1000, 10))
print(VGG16)
# 删除classifier的某层
del VGG16.classifier[7]  # 把刚增加的那层删掉
print(VGG16)
# 修改classifier的某层
VGG16.classifier[6] = nn.Linear(4096, 10)
print(VGG16)
