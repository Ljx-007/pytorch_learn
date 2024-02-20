import torch
import torch.nn as nn

# L1loss损失函数
x = torch.tensor((1, 1, 3), dtype=torch.float32)
y = torch.tensor((1, 3, 3), dtype=torch.float32)
# L1loss的计算方法为相减取绝对值后相加，然后再求平均，如果reduction='sum'，则相减取绝对值后再相加
l1_loss = nn.L1Loss(reduction='mean')
result = l1_loss(x, y)
print(result)

# MSELoss损失函数——均方误差
# MSEloss的计算方法为相减后平方再相加，最后根据reduction为mean还是sum来求平均或相加
mse_loss = nn.MSELoss()
result_mse = mse_loss(x, y)
print(result_mse)
# MSEloss和L1loss都要求输出与输入维度相同

# CrossentropyLoss交叉熵损失
# 交叉熵损失在训练多分类问题时很有用，计算方法为：
# 1.先对得到的yhat进行softmax计算 ->nn.Softmax(dim=要softmax的维度)
# 2.然后对softmax后的yhat进行负对数运算，因为softmax的结果都在0，1之间，取对数后都为负数，所以要再加个负号使其变正 ->torch.log（）
# 3.再根据label从yhat中挑出对应的值，取平均或相加，由reduction=’mean‘或sum决定 ->nn.NLLLoss()
# 1,2可以用nn.LogSoftmax()合并，然后传参给nn.NLLLoss()
# 这两步可以再合并，变成nn.CrossentropyLoss(),即nn.CrossentropyLoss()=nn.NLLLoss(nn.logSoftmax(),target)
x = torch.tensor([[0.7, 0.2, 0.3, 0.2],
                  [0.4, 0.2, 0.4, 0.6],
                  [0.4, 0.2, 0.5, 0.1]])  # 用x来假设yhat的结果，一个batch有三个样本，batch_size=3，一个样本有4类
y = torch.tensor([1, 2, 2])
x_softmax = nn.Softmax(dim=1)(x)  # 以行为单位做softmax
x_log_softmax = torch.log(x_softmax)
# 使用LogSoftmax合并softmax和log步骤
x_logsoftmax = nn.LogSoftmax(dim=1)(x)
# 验证答案是否相同
print(x_logsoftmax)
print(x_log_softmax)
# 验证nn.CrossentropyLoss()=nn.NLLLoss(nn.logSoftmax(),target)，这里都是默认reduction=“mean”
loss = nn.NLLLoss()(x_log_softmax, y)
loss_crossentropy = nn.CrossEntropyLoss()(x, y)
print(loss_crossentropy)
print(loss)
