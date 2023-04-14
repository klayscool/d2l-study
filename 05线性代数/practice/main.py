import torch

# 1 证明一个矩阵A的转置的转置是A
print('1 证明一个矩阵A的转置的转置是A')
A = torch.ones(12).reshape(4, 3)
print(A == A.T.T)

# 2 给出两个矩阵A和B，证明"它们转置的和"等于"它们和的转置"
print()
print('2 给出两个矩阵A和B，证明"它们转置的和"等于"它们和的转置"')
A = torch.ones(12).reshape(3, 4)
B = torch.tensor(([1, 2, 3, 4], [7, 9, 5, 2], [0, 1, 3, 2]))
t_s = A.T + B.T
s_t = (A + B).T
print(t_s == s_t)

# 4 本节中定义了状态为(2, 3, 4)的张量X。len(X)的输出结果是什么？
print()
print('4 本节中定义了状态为(2, 3, 4)的张量X。len(X)的输出结果是什么？')
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(len(X))
X2 = torch.arange(100).reshape(4, 5, 5)
print(X2)
print(len(X2))

# 6 运行A/A.sum(axis=1),看看会发生什么。请分析一下原因。
print()
print('6 运行A/A.sum(axis=1),看看会发生什么。请分析一下原因。')
A = torch.tensor(([1, 2, 3], [4, 5, 6], [7, 8, 9], [10 ,11, 12]))
B = A.sum(axis=0)
print("A:", A)
print("B:", B)
print("B.shape", B.shape)
print("A / B", A / B)

print()
C = A.sum(axis=1)
print("A.T", A.T)
print("C", C)
print("C.shape", C.shape)
print("A.T / C", A.T / C)

# 7 考虑一个形状为(2, 3, 4)的张量，再轴0、1、2上的求和输出是什么形状？
print()
print('7 考虑一个形状为(2, 3, 4)的张量，再轴0、1、2上的求和输出是什么形状？')
Y = torch.ones(24).reshape(2, 3, 4)
print(Y)
print(Y.sum(axis=0))
print(Y.sum(axis=1))
print(Y.sum(axis=2))

# 8 为linalg.norm函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量，这个函数计算后会得到什么？
print()
print('8 为linalg.norm函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量，这个函数计算后会得到什么？')
Z = torch.ones(2, 4, 9)
print(Z)
print(torch.linalg.norm(Z))