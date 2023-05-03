import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# 一、读取数据
print(">>> 1 Read Data")

# 读取数据到pandas中
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 将有用数据读取为张量
tr_d = train_data.iloc[:, 4:-1].drop(columns=['Laundry features', 'Appliances included', 'City', 'Parking features', 'Cooling features', 'Heating features', 'Flooring', 'Elementary School', 'Middle School', 'High School', 'Heating', 'Cooling', 'Parking', 'Region', 'Lot'])
print("tr_d.shape:", tr_d.shape)

te_d = test_data.iloc[:, 3:-1].drop(columns=['Laundry features', 'Appliances included', 'City', 'Parking features', 'Cooling features', 'Heating features', 'Flooring', 'Elementary School', 'Middle School', 'High School', 'Heating', 'Cooling', 'Parking', 'Region', 'Lot'])
print("te_d.shape:", te_d.shape)

# Bedrooms这列中有很多字符串项，导致GPU OOM，所以这里先将这7k行数据删掉
tr_d = tr_d.drop(tr_d[tr_d['Bedrooms'].str.contains('[a-zA-Z]', na=False)].index)
te_d = te_d.drop(tr_d[tr_d['Bedrooms'].str.contains('[a-zA-Z]', na=False)].index)
print("tr_d.shape:", tr_d.shape)
print("te_d.shape:", te_d.shape)

# 将数据统一，方便预处理
all_features = pd.concat((tr_d, te_d))
print('all_features.notPre.shape:', all_features.shape)

# 二、数据预处理
print(">>> 2 Preprocessing Data")

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print('all_features.afterPre.shape:', all_features.shape)

# 获取处理后的训练集和测试集
n_train = tr_d.shape[0]
print('n_train.values:', n_train)
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32, device=try_gpu())
print("train_features.values:", train_features)
print('train_features.shape:', train_features.shape)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32, device=try_gpu())
print("test_features.values:", test_features)
print('test_features.shape:', test_features.shape)

# 获取训练集的标签(同样需要将Bedrooms这列中有字符串的项过滤掉)
train_labels = torch.tensor(train_data.drop(train_data[train_data['Bedrooms'].str.contains('[a-zA-Z]', na=False)].index)['Sold Price'].values.reshape(-1, 1), dtype=torch.float32, device=try_gpu())
print("train_labels.values:", train_labels)
print('train_labels.shape:', train_labels.shape)

# 三 训练
loss = nn.MSELoss()

in_features = train_features.shape[1]
print("in_features.values:", in_features)

print(">>> 3 Train")

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 100),
                        nn.ReLU(),
                        # nn.Dropout(0.1),
                        nn.Linear(100, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1)).to(device=try_gpu())
    return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        t_l = log_rmse(net, train_features, train_labels)
        # 将每个epoch的损失打印出来
        print(f'epoch {epoch + 1}, loss {t_l:f}')

        train_ls.append(t_l)

        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# 四 K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# 五 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.5, 0.1, 200

train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: '
      f'平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

# 六 提交Kaggle预测
print(">>> 4 Export Data")
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().cpu().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)

if __name__=="__main__":
    d2l.plt.show()
    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)