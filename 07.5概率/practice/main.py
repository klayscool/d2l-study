import multiprocessing

import torch
from torch.distributions import multinomial
# 1 进行m=500组实验，每组抽取n=10个样本。改变m和n的值，观察和分析实验结果。
print('1 进行m=500组实验，每组抽取n=10个样本。改变m和n的值，观察和分析实验结果。')

n = torch.ones(10)
print('n', n)
m = 500
print(multinomial.Multinomial(m, n).sample())