import torch
from IPython import display
from d2l import torch as d2l
import time

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# print("W.shape", W.shape)
b = torch.zeros(num_outputs, requires_grad=True)

# 2 定义softmax操作
def softmax(X):
    # print("Softmax X.shape:", X.shape)
    # print("X:", X)

    X_exp = torch.exp(X)
    # print("Softmax X_exp.shape:", X_exp.shape)
    # print("X_exp:", X_exp)

    partition = X_exp.sum(1, keepdim=True)
    # print("Softmax partition.shape:", partition.shape)
    # print("partition", partition)

    r = X_exp / partition
    # print("Softmax X_exp/partition .shape:", r.shape)

    # print()
    # print("-------------------------------------------------")
    # print()

    return r

# 3 定义模型
def net(X):
    # print("X.reshape((-1, W.shape[0]) shape:", X.reshape((-1, W.shape[0])).shape)
    # print("torch.matmul(X.reshape((-1, W.shape[0])), W) shape:", torch.matmul(X.reshape((-1, W.shape[0])), W).shape)
    # print("torch.matmul(X.reshape((-1, W.shape[0])), W) + b shape:", (torch.matmul(X.reshape((-1, W.shape[0])), W) + b).shape)

    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 4 定义损失函数
def cross_entropy(y_hat, y):

    print()
    print("--------------")
    print("y_hat.value:", y_hat)
    print("y.value:", y)
    print("range(len(y_hat)).value:", range(len(y_hat)))
    print("y_hat[range(len(y_hat)), y].value:", y_hat[range(len(y_hat)), y])
    print("-torch.log(y_hat[range(len(y_hat)), y]).value:", (-torch.log(y_hat[range(len(y_hat)), y])))
    print("-torch.log(y_hat[range(len(y_hat)), y]).size:", (-torch.log(y_hat[range(len(y_hat)), y])).shape)
    print("--------------")
    print()

    return -torch.log(y_hat[range(len(y_hat)), y])

# 5 分类精度
def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""

    # print("accuracy.y_hat.value:", y_hat)
    # print("accuracy.y_hat.shape:", y_hat.shape)
    # print("accuracy.y.value:", y)
    # print("accuracy.y.shape:", y.shape)

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)

    # print("accuracy.(y.dtype).value:", y.dtype)
    #
    # print("accuracy.y_hat.type(y.dtype).value:", y_hat.type(y.dtype))
    # print("accuracy.y_hat.type(y.dtype).shape:", y_hat.type(y.dtype).shape)

    cmp = y_hat.type(y.dtype) == y
    # print("accuracy.cmp.value:", cmp)
    # print("accuracy.cmp.type(y.dtype).value:", cmp.type(y.dtype))
    return float(cmp.type(y.dtype).sum())

class Accumulator: #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter): #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    # print("metric.value:", metric)
    # print("metric.shape:", metric.shape)
    return metric[0] / metric[1]

# 6 训练
def train_epoch_ch3(net, train_iter, loss, updater): #@save
    """训练模型一轮（定义见第三章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数

        # print("Initial X.shape:", X.shape)
        # print("Initial X value:", X)

        y_hat = net(X)

        # print("y_hat.shape:", y_hat.shape)
        # print("y_hat.value:", y_hat)
        # print("y_hat.sum(1).value:", y_hat.sum(1, keepdim=True))
        print("y.shape:", y)
        print("y.value:", y)


        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            print("train_epoch_ch3.l.value:", l)
            print("train_epoch_ch3.l.shape:", l.shape)
            print("train_epoch_ch3.l.sum().value:", l.sum())
            print("train_epoch_ch3.l.sum().shape:", l.sum().shape)

            l.sum().backward()
            updater(X.shape[0])

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    # print("train_epoch_ch3.metric.value:", metric)
    # print("metric[0] / metric[2]", metric[0] / metric[2])
    # print("metric[1] / metric[2]", metric[1] / metric[2])

    print("Start sleep 3600s.")
    time.sleep(3600)

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        d2l.plt.pause(0.01)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == '__main__':
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()
    predict_ch3(net, test_iter)