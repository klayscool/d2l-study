import torch

# 1 假设我们相对函数y = 2xTx关于列向量x求导
print('1 假设我们相对函数y = 2xTx关于列向量x求导')
x = torch.arange(4.0)
print(x)

# 2 在我们计算y关于x的梯度之前，我们需要一个地方来存储梯度
print()
print('2 在我们计算y关于x的梯度之前，我们需要一个地方来存储梯度')
x.requires_grad_(True)
print(x.grad)

# 3 现在让我们计算y
print()
print('3 现在让我们计算y')
y = 2 * torch.dot(x, x)
print(y)

# 4 通过调用反向传播函数来自动计算y关于x每个分量的梯度
print()
print('4 通过调用反向传播函数来自动计算y关于x每个分量的梯度')
y.backward()
print(x.grad)

print(x.grad == 4 * x)

# 5 现在让我们计算x的另一个函数
print()
print('5 现在让我们计算x的另一个函数')
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 6 深度学习中，我们的目的不是计算微分矩阵，而是批量中每个样本单独计算的偏导数之和
print()
print('6 深度学习中，我们的目的不是计算微分矩阵，而是批量中每个样本单独计算的偏导数之和')
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

# 8 将某些计算移动到记录的计算图之外
print()
print('8 将某些计算移动到记录的计算图之外')
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)