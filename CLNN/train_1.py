from dataload import BraTs2020
import torch.nn as nn
import torch
import os
import pandas as pd
# from classification_net import classify_net
from prettytable import PrettyTable
# from VGG163d import VGG16Attention
# from VGG16LSTM_A import VGG16
from cnn_2 import classify_net
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import show_info
import time
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle
from sklearn.ensemble import RandomForestClassifier


def get_time_code():
    time_tuple = time.localtime(time.time())
    time_code = str()
    for i in range(1, 5):
        time_code += str(time_tuple[i])
    return time_code


# ----------------------------------------------------------------
# cuda
# ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

device_ids = list(range(torch.cuda.device_count()))

# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 设置基本超参
# ----------------------------------------------------------------
epochs = 50
lr = ( 1e-4 ) * 5
batch_size = 2
num_workers = 2
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 设置数据路径
# ----------------------------------------------------------------
path_dataset = "../bin/brats_npy/brats2020_3D_160"
path_trainval_txt = ""
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# 加载数据
# ----------------------------------------------------------------
with open(os.path.join(path_trainval_txt, "name_txt/train_name.txt"), "r") as f:
    train_lines = f.readlines()
with open(os.path.join(path_trainval_txt, "name_txt/val_name.txt"), "r") as f:
    val_lines = f.readlines()
train_dataset = BraTs2020(path_dataset, train_lines)
train_num = len(train_dataset)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=num_workers,
                          pin_memory=True, drop_last=False)
val_dataset = BraTs2020(path_dataset, val_lines)
val_num = len(val_dataset)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=2, num_workers=num_workers,
                        pin_memory=True,
                        drop_last=False)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# show_and_get_log_file
# ----------------------------------------------------------------
tb = PrettyTable(['parameters', 'V'])
tb.add_row(["{}".format(show_info.varname(device)), device])
tb.add_row(["{}".format(show_info.varname(device_ids)), device_ids])
tb.add_row(["{}".format(show_info.varname(epochs)), epochs])
tb.add_row(["{}".format(show_info.varname(lr)), lr])
tb.add_row(["{}".format(show_info.varname(batch_size)), batch_size])
tb.add_row(["{}".format(show_info.varname(path_dataset)), path_dataset])
tb.add_row(["{}".format(show_info.varname(path_trainval_txt)), path_trainval_txt])
tb.add_row(["{}".format(show_info.varname(train_num)), train_num])
tb.add_row(["{}".format(show_info.varname(val_num)), val_num])
tb.add_row(["{}".format(show_info.varname(num_workers)), num_workers])
print(tb)


def write_acc_or_loss(txt, epoch, value):
    txt.write('{}_{}'.format(str(epoch + 1), value))
    txt.write('\n')


# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 加载模型
# ----------------------------------------------------------------
print("model loading...")
model = classify_net()
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)
# ----------------------------------------------------------------


if __name__ == '__main__':
    path_log = 'log/{}_{}_{}_{}'.format('cnn3_c',get_time_code(),'17','2020')
    if not os.path.exists(path_log):
        os.mkdir(path_log)

    print("Starting training...")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 记录网络的准确率
    best_acc = 0.0
    # 记录svm的acc
    best_acc_svm = 0.0
    # 记录RF的acc
    best_acc_RF = 0.0
    # 模型转换成训练模式
    model_train = model.train()
    # 训练的准确率
    acc_train = 0.0
    # 定义SVM深度特征的key
    key = ['feature_{}'.format(i) for i in range(1000)]
    key.insert(0, 'label')
    # 记录loss,acc
    log_train_loss = open(os.path.join(path_log, 'train_loss.txt'), 'w+')
    log_train_acc = open(os.path.join(path_log, 'train_acc.txt'), 'w+')
    log_val_loss = open(os.path.join(path_log, 'val_loss.txt'), 'w+')
    log_val_acc = open(os.path.join(path_log, 'val_acc.txt'), 'w+')

    for epoch in range(epochs):
        acc_train = 0.0
        df = pd.DataFrame(columns=key)
        df_val = pd.DataFrame(columns=key)
        # model.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, p_labels,labels = data
            logits, x = model_train(images.to(device))
            loss = loss_function(logits, labels.to(device))
            predict_y = torch.max(logits, dim=1)[1]
            acc_train += (predict_y == labels.to(device)).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch:{}train loss: {:^3.0f}%[{}->{}]{:.3f}".format(epoch + 1, int(rate * 100), a, b, loss),
                  end="")
        print()

        # ----------------------------------------------------------------

        """
        训练classify
        """
        # ----------------------------------------------------------------


        # ----------------------------------------------------------------
        print("kaishiyanzheng")
        """
        验证
        """
        # ----------------------------------------------------------------
        # validate
        model_train.eval()
        acc_val = 0.0  # accumulate accurate number / epoch
        test_acc = 0.0
        val_loss_sum = 0.0
        with torch.no_grad():
            for index_val, val_data in enumerate(val_loader):
                val_images,val_p_labels ,val_labels = val_data
                outputs, x = model_train(val_images.to(device))
                # 计算验证loss
                val_loss = loss_function(outputs, val_labels.to(device))
                val_loss_sum += val_loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc_val += (predict_y == val_labels.to(device)).sum().item()
        print(acc_val)
        val_accurate = acc_val / val_num
        train_accurate = acc_train / train_num
        # log
        write_acc_or_loss(log_train_loss, epoch, running_loss / step)
        write_acc_or_loss(log_val_loss, epoch, val_loss_sum / index_val)
        write_acc_or_loss(log_train_acc, epoch, train_accurate)
        write_acc_or_loss(log_val_acc, epoch, val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            save_path = os.path.join(path_log, "epoch_%d_%s_%.3f.pth" % ( epoch + 1, 'CNNAt', best_acc))
            torch.save(model_train.state_dict(), save_path)


        print(
            '[epoch %d/%d] train_loss: %.3f  val_accuracy: %.3f train_accuracy: %.3f  ' %
            (
                epoch + 1, epochs, running_loss / step, val_accurate, train_accurate,
                ))

        # ----------------------------------------------------------------
    log_train_loss.close()
    log_train_acc.close()
    log_val_loss.close()
    log_val_acc.close()
    print('Finished Training')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# # ...（定义模型等的代码，与前面相同）
#
# # 定义Curriculum Learning训练函数
# def curriculum_learning_train(model, dataloader, criterion, optimizer, num_epochs=5):
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#
#         # 逐步增加难度，这里简单示范使用全部数据
#         for i, (inputs, labels) in enumerate(dataloader):
#             inputs, labels = inputs.cuda(), labels.cuda()
#
#             # 逐步增加难度的逻辑
#             if epoch < 2:
#                 # 简单阶段：使用清晰的图像
#                 pass
#             elif epoch < 4:
#                 # 中等阶段：引入轻微遮挡或者背景噪声
#                 inputs = add_noise(inputs)
#             else:
#                 # 困难阶段：引入更复杂的情况，例如模糊图像
#                 inputs = add_blur(inputs)
#
#             optimizer.zero_grad()
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             if (i + 1) % 100 == 0:
#                 print(f"Iteration {i + 1}, Loss: {loss.item()}")
#
# # 添加背景噪声的简单示例
# def add_noise(inputs):
#     noise = torch.randn_like(inputs) * 0.1  # 标准差为0.1的正态分布噪声
#     inputs = inputs + noise
#     return inputs
#
# # 添加模糊的简单示例
# def add_blur(inputs):
#     # 使用PyTorch的transforms进行模糊处理
#     blur_transform = transforms.Compose([transforms.GaussianBlur(kernel_size=3)])
#     inputs = blur_transform(inputs)
#     return inputs
#
# # ...（加载数据集等的代码，与前面相同）
#
# # 使用Curriculum Learning训练模型
# curriculum_learning_train(model, train_dataloader, criterion, optimizer, num_epochs=5)
