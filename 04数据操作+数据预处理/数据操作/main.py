# 1 首先，我们导入torch。请注意，虽然它被称为PyTorch，但我们应该导入torch而不是pytorch。
import torch

# 2 张量表示一个数值组成的数组，这个数组可能有多个纬度
print("-2-")
x = torch.arange(12)
print(x)

# 3 我们可以通过张量的shape属性来访问张量的形状和张量中元素的总数
print("-3-")
x1 = x.shape
print(x1)
x2 = x.numel()
print(x2)

# 5 要改变一个张量的形状而不改变元素数量和元素值，我们可以调用reshape函数。
print("-5-")
X = x.reshape(3, 4)
print(X)

# 6 使用全0、全1、其他常量或者从特定分布中随机采样的数字
print("-6-")
x3 = torch.zeros((2, 3, 4))
print(x3)

x4 = torch.ones((2, 3, 4))
print(x4)

# 9 通过提供包含数值的Python列表（或嵌套列表）来为所需张量中的每个元素赋予确定值
print("-9-")
x5 = torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]]).shape
print(x5)

# 10 常见的标准算术运算符（+、-、*、/和**）都可以被升级为按元素运算
print("-10-")
x6 = torch.tensor([1.0, 2, 4, 8])
y6 = torch.tensor([2, 2, 2, 2])
z1 = x6 + y6
z2 = x6 - y6
z3 = x6 * y6
z4 = x6 / y6
z5 = x6**y6
print(z1)
print(z2)
print(z3)
print(z4)
print(z5)

# 11 按元素方式应用更多的计算
print("-11-")
x7 = torch.exp(x6)
print(x7)

# 12 我们也可以把多个张量连结在一起
print("-12-")
x8 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y8 = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((x8, y8), dim=0))
print(torch.cat((x8, y8), dim=1))

# 13 通过逻辑运算符构建二元张量
print("-13-")
print(x8 == y8)

# 14 对张量中的所有元素进行求和会产生一个只有一个元素的张量。
print("-14-")
print(x8.sum())

# 15 即使形状不同，我们依然可以通过调用广播机制（broadcasting mechanism）来执行按元素操作
print("-15-")
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

# 17 可以用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素
print("-17-")
x8 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(x8)
print(x8[-1])
print(x8[1:3])
print(x8[1:3, 2:4])

# 18 初读取外，我们还可以通过指定索引来将元素写入矩阵
print("-18-")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)
X[1, 2] = 9
print(X)

# 19 为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值。
print("-19-")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
X[0:2, :] = 12
print(X)

# 20 运行一些操作可能会导致为新结果分配内存
print("-20-")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)
before = id(Y)
Y = Y + X
print(id(Y) == before)

# 21 执行原地操作
print("-21-")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Z = torch.zeros_like(Y)
print('id(Z)', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# 22 如果在后续计算中没有重复使用X，我们也可以使用X[:] = X + Y或X += Y来减少操作的内存开销。
print("-22-")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
before = id(X)
X += Y
print(id(X) == before)

# 23 转换为NumPy张量
print("-23-")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

# 24 将大小为1的张量转换为Python标量
print("---")
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))