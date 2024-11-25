import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Func
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from utils_.data_prepare import data_pre
import os


class net_radiomics(nn.Module):
    def __init__(self, input_num, output_num):
        super(net_radiomics, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_num, 116),
            nn.ReLU(),
            nn.Linear(116, 58),
            nn.ReLU(),
            nn.Linear(58, 9),
            nn.ReLU(),
            nn.Linear(9, 3),

            # nn.Softmax()
        )

    def forward(self, input):
        return self.net(input)


def data_load(data):
    '''

    :param data: dataframe,第一列为label
    :return:Tensor，x,y
    '''

    data = data_pre(data).prepare()
    x = data[data.columns[1:]]
    y = data["label"]
    x = torch.from_numpy(np.array(x, dtype=np.float32))
    y = torch.from_numpy(np.array(y))

    return x, y


path_data = "bin/csv_data/flair-re_mask-1-test.csv"
path_model_save = "./bin/model_nn/"
epoch = 50000
epoch_save = epoch - 50
lr = 0.01
num_label = 3

if __name__ == '__main__':

    data = pd.read_csv(path_data)

    train_txt = "./bin/feature_name/name_txt/train.txt"
    val_txt = "./bin/feature_name/name_txt/val.txt"

    file_train = open(train_txt, "r")
    file_val = open(val_txt, "r")

    train_list = []
    val_list = []

    for i in file_train.read().split("\n")[:-2]:
        train_list.append(int(i))
    for i in file_val.read().split("\n")[:-2]:
        val_list.append(int(i))
    data_train = data.iloc[train_list]
    train_x, train_y = data_load(data_train)
    data_val = data.iloc[val_list]
    val_x, val_y = data_load(data_val)
    train_row, train_ = train_x.shape
    val_row,val_ = val_x.shape

    # 调用网络模型
    net = net_radiomics(input_num=train_, output_num=num_label)
    net = net.cuda()

    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    folder = os.path.exists(path_model_save)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path_model_save)  # makedirs 创建文件时如果路径不存在会创建这个路径
    for i in range(epoch):
        print('Epoch：{}/{}'.format(i + 1, epoch))
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        output_train = net(train_x)  # 加载训练集，输出预测值net.forward(x),其中forward被隐藏
        loss_train = loss_func(output_train, train_y)  # loss计算，同上，forward被瘾藏
        print('train_loss:', loss_train)
        optimizer.zero_grad()  # 清空更新残余
        loss_train.backward()  # 反向传播，计算参数更新
        optimizer.step()  # 挂载更新的参数值


        val_x = val_x.cuda()
        val_y = val_y.cuda()
        output_val = net(val_x)
        loss_val = loss_func(output_val, val_y)
        print('val loss:', loss_val)
        prediction = torch.max(Func.softmax(output_val), 1)[1]  # 1表示维度1，列，[0]表示概率值，[1]表示标签
        pred_y = prediction.data.cpu().numpy()
        target_y = val_y.data.cpu().numpy()
        accuracy = sum(pred_y == target_y) / float(val_row)  # 预测中有多少和真实值一样
        print("val acc:", accuracy)
        if i > epoch_save:
            torch.save(net.state_dict(), path_model_save + 'ep%03d-loss%.3f.pth' % ((i + 1), loss_val))
