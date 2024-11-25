from MSANN.dataload import BraTs
import torch.nn as nn
import torch
import os
from prettytable import PrettyTable
from MSANN.CNN_A_NEW import classify_net
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import show_info
import time






def del_model(dir_log):
    list_files = os.listdir(dir_log)
    for f in list_files:
        if f.endswith('.pth'):
            os.remove(os.path.join(dir_log,f))
def write_acc_or_loss(txt, epoch, value):
    txt.write('{}_{}'.format(str(epoch + 1), value))
    txt.write('\n')

def train(path_log_MSANN,dataset):
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
    epochs = 20
    lr = (1e-4) *5
    batch_size = 2
    num_workers = 2
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # 设置数据路径
    # ----------------------------------------------------------------
    # ROOT directory
    path_home = os.path.dirname(__file__)
    path_boot = os.path.dirname(path_home)

    path_dataset = os.path.join(path_boot,f"bin/brats_npy/brats2020_3D_160")

    path_trainval_txt = os.path.join(path_boot, f"bin/nametxt/brats{dataset}")
    # ----------------------------------------------------------------
    # 加载数据
    # ----------------------------------------------------------------
    with open(os.path.join(path_trainval_txt, "train_name.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(path_trainval_txt, "val_name.txt"), "r") as f:
        val_lines = f.readlines()
    train_dataset = BraTs(path_dataset, train_lines)
    train_num = len(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, drop_last=False)
    val_dataset = BraTs(path_dataset, val_lines)
    val_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # show_and_get_log_file
    # ----------------------------------------------------------------
    tb = PrettyTable(['parameters', 'V'])
    tb.add_row(["{}".format(show_info.varname(device)), device])
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

    # ----------------------------------------------------------------
    # 加载模型
    # ----------------------------------------------------------------
    print("model loading...")
    model = classify_net()
    # model = nn.DataParallel(model, device_ids=device_ids)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
# ----------------------------------------------------------------------------------------------------------


    path_log = path_log_MSANN
    if not os.path.exists(path_log):
        os.mkdir(path_log)
    print("Starting training...")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 记录网络的准确率
    best_acc = 0.0
    # 模型转换成训练模式
    model_train = model.train()
    # 训练的准确率
    acc_train = 0.0

    # 记录loss,acc
    log_train_loss = open(os.path.join(path_log, 'train_loss.txt'), 'w+')
    log_train_acc = open(os.path.join(path_log, 'train_acc.txt'), 'w+')
    log_val_loss = open(os.path.join(path_log, 'val_loss.txt'), 'w+')
    log_val_acc = open(os.path.join(path_log, 'val_acc.txt'), 'w+')

    for epoch in range(epochs):

        acc_train = 0.0

        # model.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
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
                val_images, val_labels = val_data
                outputs, x = model_train(val_images.to(device))
                # 计算验证loss
                val_loss = loss_function(outputs, val_labels.to(device))
                val_loss_sum += val_loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc_val += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc_val / val_num
        train_accurate = acc_train / train_num
        # log
        write_acc_or_loss(log_train_loss, epoch, running_loss / step)
        write_acc_or_loss(log_val_loss, epoch, val_loss_sum / index_val)
        write_acc_or_loss(log_train_acc, epoch, train_accurate)
        write_acc_or_loss(log_val_acc, epoch, val_accurate)

        if val_accurate > best_acc:
            del_model(path_log)
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
