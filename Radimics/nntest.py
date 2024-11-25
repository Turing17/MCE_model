import torch
import torch.nn.functional as Func  # 激励函数都在这
import matplotlib.pyplot as plt

# 假数据
n_data = torch.ones(100, 2)  # 数据的基本形态
x0 = torch.normal(2 * n_data, 1)  # 类型0 x data (tensor), shape=(100, 2) norm归一化
y0 = torch.zeros(100)  # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2 * n_data, 1)  # 类型1 x data (tensor), shape=(100, 2) norm归一化
y1 = torch.ones(100)  # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating #按维数0拼接，行
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # LongTensor = 64-bit integer
print('x', y)
print('y', y.shape)


# 画图
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 Module 的 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature,
                                      n_hidden)  # 隐藏层线性输出, type(hidden) = torch.nn.modules.linear.Linear(一个类)
        self.output = torch.nn.Linear(n_hidden,
                                      n_output)  # 输出层线性输出, type(predict) = torch.nn.modules.linear.Linear(一个类)

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = Func.relu(self.hidden(x))  # 激励函数(隐藏层的线性值) self.hidden.forward(x), 其中forward被隐藏，因为使用了继承，父类中有@内置
        x = self.output(x)  # 输出值 self.predict.forward(x), 其中forward被隐藏
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)

print(net)  # net 的结构

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()  # 预测值和真实值的误差计算公式 (交叉熵), type(torch.nn.CrossEntropyLoss()) = torch.nn.modules.loss.CrossEntropyLoss(一个类)

plt.ion()  # 画图
plt.show()

for t in range(100):
    output = net(x)  # 喂给 net 训练数据 x, 输出预测值 net.forward(x), 其中forward被隐藏
    loss = loss_func(output, y)  # 计算两者的误差 loss_func.forward(prediction, y), 其中forward被隐藏
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(Func.softmax(output), 1)[1]  # 1表示维度1，列，[0]表示概率值，[1]表示标签
        print(prediction)
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
