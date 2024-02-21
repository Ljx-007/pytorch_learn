import torch
import torchvision.models

VGG16 = torchvision.models.vgg16()
# 保存方法1：模型结构+参数
torch.save(VGG16, "VGG16.pth")  # 保存的对象为VGG16，路径为当前文件下的VGG16,后缀通常用pth，pth通常用于保存多个模型，pt为单个模型
# 加载模型1
vgg = torch.load("VGG16.pth")
print(vgg)
# 保存方法2：模型参数
torch.save(VGG16.state_dict(), "dict_VGG.pth")  # 保存模型的状态，以字典形式保存模型参数
diction = torch.load("dict_VGG.pth")
print(diction)  # 打印出来都是模型的参数，以字典形式呈现
# 加载模型2
VGG16.load_state_dict(torch.load("dict_VGG.pth"))  # VGG16这个模型加载了dict_VGG.pth中保存的参数
