from dataload import BraTs2020
import torch.nn as nn
import torch
import os
import pandas as pd
# from classification_net import classify_net
from prettytable import PrettyTable
# from VGG163d import VGG16Attention
# from VGG16LSTM_A import VGG16
# from CNN import DCN
from CNN_A_NEW import classify_net
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
lr = 1e-3
batch_size = 2
num_workers = 2
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 设置数据路径
# ----------------------------------------------------------------
path_dataset = "../bin/brats_npy/brats2020_3D_160"
path_trainval_txt = "../bin/nametxt/brats2018"
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# 加载数据
# ----------------------------------------------------------------
with open(os.path.join(path_trainval_txt, "train_name.txt"), "r") as f:
    train_lines = f.readlines()
with open(os.path.join(path_trainval_txt, "val_name.txt"), "r") as f:
    val_lines = f.readlines()
train_dataset = BraTs2020(path_dataset, train_lines)
train_num = len(train_dataset)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=True, drop_last=False)
val_dataset = BraTs2020(path_dataset, val_lines)
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


def write_acc_or_loss(txt, epoch, value):
    txt.write('{}_{}'.format(str(epoch + 1), value))
    txt.write('\n')


# ----------------------------------------------------------------

# ----------------------------------------------------------------
# 加载模型
# ----------------------------------------------------------------
print("model loading...")
model = classify_net()
# model = nn.DataParallel(model, device_ids=device_ids)
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)
# ----------------------------------------------------------------


if __name__ == '__main__':
    path_log = '../cnnanew/log/{}_{}_{}'.format(get_time_code(),'6','2018')
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
            images, labels = data
            logits, x = model_train(images.to(device))
            # 将fc的1000层转成list
            y_list = labels.tolist()
            X_list = x.tolist()
            for i in range(images.size()[0]):
                temp = X_list[i]
                temp.insert(0, y_list[i])
                if i == 0:
                    df.loc[step * 2] = temp
                else:
                    df.loc[(step * 2) + 1] = temp
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
        X_train = df[df.columns[1:]]
        # print(X_train)
        y_train = df['label']
        # svm
        clf = svm.SVC(kernel='sigmoid')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        train_accuracy_svm = accuracy_score(y_train, y_pred)
        # rf
        rf = RandomForestClassifier(n_estimators=10000, random_state=7)
        rf.fit(X_train, y_train)
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
                # 将验证集的转成df
                X_list = x.tolist()
                y_list = val_labels.tolist()
                temp = X_list[0]
                temp.insert(0, y_list[0])

                # 写入df
                df_val.loc[index_val] = temp

                predict_y = torch.max(outputs, dim=1)[1]
                acc_val += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc_val / val_num
        train_accurate = acc_train / train_num

        # SVM测试
        X_val = df_val[df_val.columns[1:]]
        y_val = df_val['label']
        y_pred = clf.predict(X_val)
        val_accuracy_svm = accuracy_score(y_val, y_pred)
        # log
        write_acc_or_loss(log_train_loss, epoch, running_loss / step)
        write_acc_or_loss(log_val_loss, epoch, val_loss_sum / index_val)
        write_acc_or_loss(log_train_acc, epoch, train_accurate)
        write_acc_or_loss(log_val_acc, epoch, val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            save_path = os.path.join(path_log, "epoch_%d_%s_%.3f.pth" % ( epoch + 1, 'CNNAt', best_acc))
            torch.save(model_train.state_dict(), save_path)
        if rf.score(X_val, y_val) > best_acc_RF:
            best_acc_RF = rf.score(X_val, y_val)
            save_path = os.path.join(path_log, "%s_epoch_%d_%s_%.3f.pth" % ('rf',epoch + 1, 'CNNAt', best_acc))
            torch.save(model_train.state_dict(), save_path)
            with open(os.path.join(path_log, 'epoch{}_rf_model_{}.pkl'.format(epoch + 1,best_acc_RF)), 'wb') as file:
                pickle.dump(clf, file)
            df.to_csv(os.path.join(path_log, 'epoch{}_rf_train_feature_{}.csv'.format(epoch + 1,best_acc_RF)),index=False)
            df_val.to_csv(os.path.join(path_log, 'epoch{}_rf_val_feature_{}.csv'.format(epoch + 1, best_acc_RF)),
                      index=False)
        if val_accuracy_svm > best_acc_svm:
            best_acc_svm = val_accuracy_svm
            save_path = os.path.join(path_log, "%s_epoch_%d_%s_%.3f.pth" % ('svm',epoch + 1, 'CNNAt', best_acc))
            torch.save(model_train.state_dict(), save_path)
            with open(os.path.join(path_log, 'epoch{}_svm_model_{}.pkl'.format(epoch + 1,best_acc_svm)), 'wb') as file:
                pickle.dump(clf, file)
            df.to_csv(os.path.join(path_log, 'epoch{}_SVM_train_feature_{}.csv'.format(epoch + 1, best_acc_svm)),
                      index=False)
            df_val.to_csv(os.path.join(path_log, 'epoch{}_SVM_val_feature_{}.csv'.format(epoch + 1, best_acc_svm)),
                          index=False)
        print(
            '[epoch %d/%d] train_loss: %.3f  val_accuracy: %.3f train_accuracy: %.3f train_accuracy_svm: %.3f val_accuracy_svm: %.3f' %
            (
                epoch + 1, epochs, running_loss / step, val_accurate, train_accurate, train_accuracy_svm,
                val_accuracy_svm))
        print('RF Accuracy train: {:.3f}'.format(rf.score(X_train, y_train)),
              'RF Accuracy val: {:.3f}'.format(rf.score(X_val, y_val)))
        # ----------------------------------------------------------------
    log_train_loss.close()
    log_train_acc.close()
    log_val_loss.close()
    log_val_acc.close()
    print('Finished Training')
