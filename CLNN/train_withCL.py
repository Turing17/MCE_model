from CLNN.dataload import BraTs2020
import torch.nn as nn
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CLNN.cnn_3 import classify_net
import time
import show_info
from prettytable import PrettyTable
import pandas as pd

def del_model(dir_log):
    list_files = os.listdir(dir_log)
    for f in list_files:
        if f.endswith('.pth'):
            os.remove(os.path.join(dir_log,f))

def get_time_code():
    time_tuple = time.localtime(time.time())
    time_code = str()
    for i in range(1, 5):
        time_code += str(time_tuple[i])
    return time_code



def write_acc_or_loss(txt, epoch, value):
    txt.write('{}_{}'.format(str(epoch + 1), value))
    txt.write('\n')


# ----------------------------------------------------------------
def train(path_log_CLNN,dataset):
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
    epochs = 135
    lr = (1e-4) * 5
    batch_size = 2
    num_workers = 2
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 设置数据路径
    # ----------------------------------------------------------------
    path_home = os.path.dirname(__file__)
    path_boot = os.path.dirname(path_home)
    path_dataset = os.path.join(path_boot,f"bin/brats_npy/brats2020_3D_160")
    path_trainval_txt = os.path.join(path_home, f"name_txt/brats{dataset}")
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 加载数据
    # ----------------------------------------------------------------
    with open(os.path.join(path_trainval_txt, f"train_name.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(path_trainval_txt, f"val_name.txt"), "r") as f:
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

    # ----------------------------------------------------------------
    # 加载模型
    # ----------------------------------------------------------------
    print("model loading...")
    model = classify_net()
    # model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    freezy_index = 50

    path_log = path_log_CLNN
    if not os.path.exists(path_log):
        os.mkdir(path_log)

    print("Starting training...")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 记录网络的准确率

    # 记录svm的acc
    best_acc_svm = 0.0
    # 记录RF的acc
    best_acc_RF = 0.0
    # 训练的准确率
    acc_train = 0.0
    # 记录loss,acc
    log_train_loss = open(os.path.join(path_log, 'train_loss.txt'), 'w+')
    log_train_acc = open(os.path.join(path_log, 'train_acc.txt'), 'w+')
    log_val_loss = open(os.path.join(path_log, 'val_loss.txt'), 'w+')
    log_val_acc = open(os.path.join(path_log, 'val_acc.txt'), 'w+')
    best_acc_p = 0.0
    best_acc_c = 0.0
    for epoch in range(epochs):
        flag = 1
        if epoch < freezy_index:

            acc_train = 0.0
            # model.train()
            running_loss = 0.0
            print('P_train>>>>>>>>')
            # for param in model.parameters():
            #     param.requires_grad = False
            for param in model.classfication_p.parameters():
                param.requires_grad = False
            for param in model.classfication_c.parameters():
                param.requires_grad = False
            for param in model.features_extract.parameters():
                param.requires_grad = True

            model.train()
            for step, data in enumerate(train_loader, start=0):
                images,without_w_mask, p_labels, labels = data
                optimizer.zero_grad()
                output = model(images.to(device),without_w_mask,flag)
                loss = loss_function(output, p_labels.to(device))
                predict_y = torch.max(output, dim=1)[1]
                acc_train += (predict_y == p_labels.to(device)).sum().item()

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
            print("kaishiyanzheng")
            """
            验证
            """
            # ----------------------------------------------------------------
            # validate
            model.eval()
            acc_val = 0.0  # accumulate accurate number / epoch
            test_acc = 0.0
            val_loss_sum = 0.0
            with torch.no_grad():
                for index_val, val_data in enumerate(val_loader):
                    val_images, without_w_mask,val_p_labels, val_labels = val_data
                    output = model(val_images.to(device),without_w_mask,flag)
                    # 计算验证loss
                    val_loss = loss_function(output, val_p_labels.to(device))
                    val_loss_sum += val_loss.item()
                    predict_y = torch.max(output, dim=1)[1]
                    acc_val += (predict_y == val_p_labels.to(device)).sum().item()
            print(acc_val)
            val_accurate = acc_val / val_num
            train_accurate = acc_train / train_num
            # log
            write_acc_or_loss(log_train_loss, epoch, running_loss / step)
            write_acc_or_loss(log_val_loss, epoch, val_loss_sum / index_val)
            write_acc_or_loss(log_train_acc, epoch, train_accurate)
            write_acc_or_loss(log_val_acc, epoch, val_accurate)

            if val_accurate > best_acc_p:
                del_model(path_log)
                best_acc_p = val_accurate
                save_path = os.path.join(path_log, "epoch_%d_%s_%.3f.pth" % (epoch + 1, 'CNNAt', best_acc_p))
                torch.save(model.state_dict(), save_path)

            print(
                '[epoch %d/%d] train_loss: %.3f  val_accuracy: %.3f train_accuracy: %.3f  ' %
                (
                    epoch + 1, epochs, running_loss / step, val_accurate, train_accurate,
                ))

            # ----------------------------------------------------------------

        elif epoch >= freezy_index:
            flag = 0
            acc_train = 0.0
            best_acc = 0.0

            # model.train()
            running_loss = 0.0

            print('C_train>>>>>>>>')
            # for param in model.parameters():
            #     param.requires_grad = False
            for param in model.features_extract.parameters():
                param.requires_grad = True
            for param in model.classfication_c.parameters():
                param.requires_grad = True
            for param in model.classfication_p.parameters():
                param.requires_grad = False

            model.train()

            for step, data in enumerate(train_loader, start=0):
                images, without_w_mask,p_labels, labels = data
                optimizer.zero_grad()
                output = model(images.to(device),without_w_mask,flag)
                loss = loss_function(output, labels.to(device))
                predict_y = torch.max(output, dim=1)[1]
                acc_train += (predict_y == labels.to(device)).sum().item()

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

            """
            验证
            """
            # ----------------------------------------------------------------
            # validate
            model.eval()
            acc_val = 0.0  # accumulate accurate number / epoch
            test_acc = 0.0
            val_loss_sum = 0.0
            with torch.no_grad():
                for index_val, val_data in enumerate(val_loader):
                    val_images,without_w_mask, val_p_labels, val_labels = val_data
                    output = model(val_images.to(device),without_w_mask,flag)
                    # 计算验证loss
                    val_loss = loss_function(output, val_labels.to(device))
                    val_loss_sum += val_loss.item()
                    predict_y = torch.max(output, dim=1)[1]
                    acc_val += (predict_y == val_labels.to(device)).sum().item()
            print(acc_val)
            val_accurate = acc_val / val_num
            train_accurate = acc_train / train_num
            # log
            write_acc_or_loss(log_train_loss, epoch, running_loss / step)
            write_acc_or_loss(log_val_loss, epoch, val_loss_sum / index_val)
            write_acc_or_loss(log_train_acc, epoch, train_accurate)
            write_acc_or_loss(log_val_acc, epoch, val_accurate)

            if val_accurate > best_acc_c:
                del_model(path_log)
                best_acc_c = val_accurate
                save_path = os.path.join(path_log, "epoch_%d_%s_%.3f.pth" % (epoch + 1, 'CLNN', best_acc_c))
                torch.save(model.state_dict(), save_path)

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



# 定义Curriculum Learning训练函数
# def curriculum_learning_train(model, dataloader, criterion, optimizer, num_epochs=50):
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#
#         # 逐步增加难度，这里简单示范使用全部数据
#         for i, (inputs, labels) in enumerate(dataloader):
#             inputs, labels = inputs.cuda(), labels.cuda()
#
#             # 逐步增加难度的逻辑
#             if epoch < 25:
#                 # 训练序列分类
#
#                 pass
#             elif 34 < epoch < 50:
#
#                 # 增加roi,进行H_L的分类
#                 inputs = add_noise(inputs)
#
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
