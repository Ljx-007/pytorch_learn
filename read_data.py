from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import numpy as np


# cv2读取图片，参数1为图片路径，参数2为图片样式
# img = cv2.imread("dataset/train/ants/0013035.jpg", 1)  # 0为灰度图，1为彩色图，无透明度，-1为彩色图有透明度
# # 展示图片
# cv2.imshow("image", img)  # 参数一为字符串，生成图片的窗口名；参数二为展示的图片的矩阵
# k = cv2.waitKey(1000)  # 窗口停留时间：1000ms
# print(img.shape)


class Mydataset(Dataset):
    # init函数通常对参数进行初始化，方便后续函数使用,init的参数就是调用Mydataset类时的参数
    def __init__(self, root_dir, label_dir):  # 有了self就是一个类的实例的属性，没有self就是类的属性，并且有了self的变量可以在这个类中当作全局变量
        # 把两个路径root_dir,label_dir变成类的全局变量
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)  # 把两个路径拼到一起
        self.img_path = os.listdir(self.path)  # 把path路径下的文件变成列表形式，列表的元素类型为字符串

    # getitem函数用来获取单个样本，对单个样本进行处理，并返回单个样本数据及标签
    def __getitem__(self, index):
        # 用索引index把图片名从列表中拿出来，这里的img_path有了self后在Mydataset这个类里就是全局变量了，所以可以拿到这个函数中来用
        img_name = self.img_path[index]
        # 把各个路径组合起来得到dataset中第一个图片的路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    # len函数用来返回数据集的大小
    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
# 调用Mydataset类时参数与init时的参数一致
ants_dataset = Mydataset(root_dir, ants_label_dir)  # 这样ants_dataset中就有了root_dir,label_dir,path,img_path这些值
bees_dataset = Mydataset(root_dir, bees_label_dir)
img, label = ants_dataset[0]  # 取出数据集中的第一个元素
img2, label2 = bees_dataset[0]
img2.show()
# 将两个数据集拼接在一起，ants在前，bees在后
train_dataset = ants_dataset + bees_dataset

# 一般label会存到一个txt文档中，对应img的文件名，所以新建两个文件夹，创建txt文档填写对应的img的标签，并且命名为img的文件名
target_dir = "ants_image"
# 定义要输出到的文件夹
out_dir = "ants_label"
# 合成图像名列表img_path
img_path = os.listdir(os.path.join(root_dir, target_dir))
for i in img_path:
    img_name = i.split('.')[0]  # 从每个图像名字中去掉.jpg。split('.')是以.为分割符，把被分割的字符串重新组成元组，这里被分割成文件名和jpg，取前面的文件名
    # 打开创建好的ants_label文件夹，往以文件名为img_name.txt的文档中写入东西，如果没有该文档则创建一个文档
    with open(os.path.join(root_dir, out_dir, img_name + ".txt"), 'w') as file:
        file.write(target_dir.split('_')[0])  # target_dir所代表的字符串被'_'分割成ants和label，写入前面的ants部分
