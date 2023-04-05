# 1 创建一个人工数据集，并存储在csv（逗号分割值）文件
import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 2 从创建的csv文件中加载原始数据集
print('-2-')
data = pd.read_csv(data_file)
print(data)

# 3 为了处理缺失的数据，典型的方法包括插值和删除，这里，我们将考虑插值
print('-3-')
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 4 对于inputs中的类别值或离散值，我们将"NaN"视为一个类别。
print('-4-')
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 5 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
print('-5-')
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)

print('-test-')
a = torch.arange(12)
print(a)
b = a.reshape((3, 4))
print(b)
b[:] = 2
print(b)
print(a)