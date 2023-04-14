import os
import pandas as pd
import torch
import numpy as np

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny_practise.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    f.write('NA,NA,140001\n')
    f.write('NA,NA,140002\n')
    f.write('NA,NA,140003\n')
    f.write('NA,NA,140004\n')
    f.write('NA,NA,140005\n')
    f.write('NA,NA,140006\n')

data = pd.read_csv(data_file)
# datafa = pd.DataFrame(data)
# l = list()
# lfd = list()
# for key, _ in datafa.iteritems():
#     isnil = (datafa[key].isnull())
#     c = 0
#     for k in isnil:
#         if k == True:
#             c += 1
#     f = [('name', key), ('c', c)]
#     l.append(c)
#     fd = dict(f)
#     lfd.append(fd)
# maxc = max(l)
# for k in lfd:
#     if k['c'] == maxc:
#         datafa2 = datafa.drop(k['name'], axis=1)
# my_np = np.array(datafa2)
# my_tensor = torch.tensor(my_np)
# print(my_tensor)

count = 0
count_max = 0
labels = ['NumRooms','Alley','Price']
for label in labels:
    count = data[label].isna().sum()
    if count > count_max:
        count_max = count
        flag = label
data_new = data.drop(flag,axis=1)
print(data_new)