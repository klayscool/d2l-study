import torch

# 1 标量由只有一个元素的张量表示
print('-1 标量由只有一个元素的张量表示-')
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y)
print(x * y)
print(x / y)
print(x**y)

# 2 你可以将向量视为标量值组成的列表
print('-2 你可以将向量视为标量值组成的列表-')
x = torch.arange(4)
print(x)

# 3 通过张量的索引来访问任一元素
print('-3 通过张量的索引来访问任一元素-')
print(x[3])
print(x[2])

# 4 访问张量的长度
print('-4 访问张量的长度-')
print(len(x))

# 5 只有一个轴的张量，形状只有一个元素
print('-5 只有一个轴的张量，形状只有一个元素-')
print(x.shape)

# 6 通过指定两个分量m和n来创建一个形状为m*n的矩阵
print('-6 通过指定两个分量m和n来创建一个形状为m*n的矩阵-')
A = torch.arange(20).reshape(5, 4)
print(A)

# 7 矩阵的转置
print('-7 矩阵的转置-')
print(A.T)

# 8 对称矩阵(symmetric matrix) A 等于其转置：A = A.T
print('-8 对称矩阵(symmetric matrix) A 等于其转置：A = A.T-')
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

# 9 就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构
print('9 就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构')
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 11 给定具有相同形状的任何两个张量，任何按元素二元运算的结果都将是相同形状的张量
print('11 给定具有相同形状的任何两个张量，任何按元素二元运算的结果都将是相同形状的张量')
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
print(A)
print(A + B)

# 12 两个矩阵的按元素乘法称为哈达玛积（Hadamard product）
print('12 两个矩阵的按元素乘法称为哈达玛积（Hadamard product）')
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)

# 14 计算其元素的和
print('14 计算其元素的和')
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())

# 15 表示任意形状张量的元素和
print('15 表示任意形状张量的元素和')
# A = torch.arange(20*2).reshape(2, 5, 4)
print(A)
print(A.shape)
print(A.sum())

# 16 指定求和汇总张量的轴
print('16 指定求和汇总张量的轴')
print(A)
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)

# 19 一个与求和相关的量是平均值（mean或average）
print()
print('19 一个与求和相关的量是平均值（mean或average）')
print(A)
print(A.mean())
print(A.sum() / A.numel())

print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])

# 21 计算总和或均值时保持轴数不变
print()
print('21 计算总和或均值时保持轴数不变')
print(A)
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

# 22 通过广播将A除以sum_A
print()
print('22 通过广播将A除以sum_A')
print(A / sum_A)

# 23 某个轴计算A元素的累积总和
print()
print('23 某个轴计算A元素的累积总和')
print(A.cumsum(axis=0))

# 24 点积是相同位置的按元素乘积的和
print()
print('24 点积是相同位置的按元素乘积的和')
y = torch.ones(4, dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x, y))

# 25 我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积
print()
print('25 我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积')
print(torch.sum(x * y))

# 26 矩阵向量积Ax是一个长度为m的列向量，其中ith元素是点积aTiX
print()
print('26 矩阵向量积Ax是一个长度为m的列向量，其中ith元素是点积aTiX')
print(A)
print(A.shape)
print(x.shape)
print(torch.mv(A, x))

# 27 我们可以将矩阵-矩阵乘法AB看作是简单的执行了m次矩阵-向量积，并将结果拼接在一起，形成一个n*m矩阵
print()
print('27 我们可以将矩阵-矩阵乘法AB看作是简单的执行了m次矩阵-向量积，并将结果拼接在一起，形成一个n*m矩阵')
B = torch.ones(4, 3)
print(A)
print(B)
print(torch.mm(A, B))

# 28 L2范数是向量元素平方和的平方根
print()
print('28 L2范数是向量元素平方和的平方根')
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# 29 L1范数，它表示为向量元素的绝对值之和
print()
print('29 L1范数，它表示为向量元素的绝对值之和')
print(torch.abs(u).sum())

# 30 矩阵的弗罗贝尼乌斯范数（Frobenius norm）是矩阵元素的平方和的平方根
print()
print('30 矩阵的弗罗贝尼乌斯范数（Frobenius norm）是矩阵元素的平方和的平方根')
print(torch.ones((4, 9)))
print(torch.sum(torch.ones(4, 9)))
print(torch.norm(torch.ones((4, 9))))