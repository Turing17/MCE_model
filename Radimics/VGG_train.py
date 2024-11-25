import pandas as pd
from VGG.utils_vgg.VGG16 import vgg16
import torch
path_data = "./bin/result_f/label1/"
path_model_save = "./bin/model_nn/"
epoch = 50000
epoch_save = epoch - 50
lr = 0.01
num_label = 3

if __name__ == '__main__':
    # 调用网络模型
    net = vgg16(input_num=train_, output_num=num_label)
    x = torch.rand(size=(8, 1, 224, 224))
    vgg16 = vgg16(nums=1000)
    out = vgg16(x)

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


    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    folder = os.path.exists(path_model_save)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path_model_save)  # makedirs 创建文件时如果路径不存在会创建这个路径
    for i in range(epoch):
        print('Epoch：{}/{}'.format(i + 1, epoch))
        output_train = net(train_x)  # 加载训练集，输出预测值net.forward(x),其中forward被隐藏
        loss_train = loss_func(output_train, train_y)  # loss计算，同上，forward被瘾藏
        print('train_loss:', loss_train)
        optimizer.zero_grad()  # 清空更新残余
        loss_train.backward()  # 反向传播，计算参数更新
        optimizer.step()  # 挂载更新的参数值

        output_val = net(val_x)
        loss_val = loss_func(output_val, val_y)
        print('val loss:', loss_val)
        prediction = torch.max(Func.softmax(output_val), 1)[1]  # 1表示维度1，列，[0]表示概率值，[1]表示标签
        pred_y = prediction.data.numpy()
        target_y = val_y.data.numpy()
        accuracy = sum(pred_y == target_y) / float(val_row)  # 预测中有多少和真实值一样
        print("val acc:", accuracy)
        if i > epoch_save:
            torch.save(net.state_dict(), path_model_save + 'ep%03d-loss%.3f.pth' % ((i + 1), loss_val))