import torch

# 2 在执行反向传播函数之后，立即再次执行它，看看会发生什么。
print('2 在执行反向传播函数之后，立即再次执行它，看看会发生什么。')
x = torch.arange(4.0)
print('x', x)

x.requires_grad_(True)
print('x.grad', x.grad)

y = 2 * torch.dot(x, x)
print('y', y)

y.backward(retain_graph=True)
print('x.gard', x.grad)
y.backward()
print('x.gard', x.grad)

# 3 在控制流的例子中，我们计算d关于a的导数，如果将变量a更改为随机向量或矩阵，会发生什么？
print()
print('3 在控制流的例子中，我们计算d关于a的导数，如果将变量a更改为随机向量或矩阵，会发生什么？')
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(3, 4), requires_grad=True)
print('a.shape:', a.shape)
print('a:', a)
d = f(a)
print('d:', d)
d.sum().backward()
print('a.grad:', a.grad)

# 4 重新设计一个求控制流梯度的例子，运行并分析结果
print()
print('4 重新设计一个求控制流梯度的例子，运行并分析结果')
import torch
x = torch.randn(size=(3, 6), requires_grad=True)
print('x:', x)
t = torch.randn(size=(6, 4), requires_grad=True)
print('t:', t)
y = 2 * torch.mm(x, t)
print('y:', y)
print('y.sum().backward():', y.sum().backward())
print('x.grad:', x.grad)
print('t.grad:', t.grad)