from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# torchvision用于计算机视觉，一般与图像相关
from PIL import Image

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# 创建一个类的实例，有了类的实例才能传参

totensor = transforms.ToTensor()  # ToTensor的作用是把一张PIL图或np数组转化为tensor型数据

# tensor数据类型是神经网络专用的数据类型，包装了许多神经网络所需要的参数如backward，grads等
# ToTensor中有个__call__函数，它的作用是把实例化好的对象当作函数，加个括号填写参数就可以使用

img_tensor = totensor(img)  # 或img_tensor=transforms.ToTensor()(img)也可以，效果相同
write = SummaryWriter("log")
write.add_image("img", img_tensor, 1)

# Normalize归一化
# output[channel] = (input[channel] - mean[channel]) / std[channel]
norm = transforms.Normalize([1, 2, 1], [0.5, 0.5, 0.5])  # 将每个通道都归一化，输入每个通道的均值mean和标准差std，这里图片有三个通道，于是设置三个值
img_norm = norm(img_tensor)

# Resize 调整PIL图像大小
print(img.size)
resize = transforms.Resize(100)  # Resize的参数需要规定调整后的size大小，如果只有一个数的话则图像最小的边等于这个数进行等比缩放
img_resize = resize(img)
print(img_resize.size)
img_resize_tensor = totensor(img_resize)
#write.add_image("img", img_resize_tensor, 2)

# Compose 对以上transforms的操作进行拼接，一步到位
compose = transforms.Compose([resize, totensor])  # 用列表的形式传参，先resize再totensor
img_resize = compose(img)

# RandomCrop 随机裁剪
random_crop = transforms.RandomCrop((100,120))
compose2 = transforms.Compose([random_crop, totensor])
# 随即裁剪
for i in range(10):
    random = compose2(img)
    write.add_image("random", random, i)
write.close()
