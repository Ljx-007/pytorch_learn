from torch.utils.tensorboard import SummaryWriter
import cv2

# SummaryWriter相当于创建一个日志log，记录其中的数据，是可视化数据的一个方法
write = SummaryWriter("log")  # log就是日志的名字，会在当前代码所在文件夹下自动创建
# # tensorboard和jupyter notebook差不多，都需要从终端输入命令来打开，打开tensorboard命令为：tensorboard --logdir=日志文件名
# for i in range(10):
#     # 将标量数据可视化
#     write.add_scalar("y=x", i, i)  # write.add_scalar的tag参数是str型，是一个图像的标题，剩下两个参数为value（y轴）和步长（x轴）
#     write.add_scalar("y=x**2",i**2,i)
# write.close()  # 关闭日志

# 刚才是写入标量，还可以写入图片
img = cv2.imread("dataset/train/bees_image/16838648_415acd9e3f.jpg", 1)
# write.add_image要求输入的图像张量是numpy型或torch.Tensor型，cv2读取的img就是numpy型
# write.add_image还默认输入的图片维度为（C,H,W)，通道在前，如果要输入通道在后的维度，则用dataformats来指定
write.add_image("image", img, 2, dataformats='HWC')
write.close()
# 通过tensorboard可以看到模型训练每个阶段的输出结果，将输出数据可视化
