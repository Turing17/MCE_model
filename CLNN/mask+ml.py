import torch
from torch.utils.data import DataLoader
from dataload import BraTs2020
import pandas as pd
import os
from tqdm import tqdm
from cnn import classify_net
from cnn_3 import classify_net as mask_net
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle
import torch.nn.init as init
from sklearn.ensemble import RandomForestClassifier
from mask_dataload import BraTs2020 as BraTs2020_mask
import csv
import numpy as np
from sklearn.model_selection import GridSearchCV
def mask_to_df():
    key = ['feature_{}'.format(i) for i in range(32000)]
    key.insert(0, 'label')
    result_df = pd.DataFrame(columns=key)
    # 读取两个CSV文件
    file1_path = 'DF/train.csv'
    file2_path = 'DF/mask_ori_cnn2.csv'
    path_train_name = 'name_txt/train_name.txt'
    # 读取CSV文件为DataFrame
    df1 = pd.read_csv(file1_path)
    image_df = df1.iloc[:, 1:]
    image_label = df1.iloc[:,0].tolist()
    # print(image_df.shape)
    mask_df = pd.read_csv(file2_path)
    with open(path_train_name,'r') as f:
        lines = f.readlines()
    for step, i in tqdm(enumerate(lines)):
        df = image_df.iloc[step]
        index = int(i.split('_')[2])
        # print('index:', index)
        mask = mask_df.iloc[index-1]
        result_row = df * mask
        result_row = result_row.tolist()
        result_row.insert(0, image_label[step])
        result_df.loc[step] = result_row
        # 将结果保存为新的CSV文件-


    result_file_path = 'DF/train_mask.csv'
    result_df.to_csv(result_file_path, index=False)
    # # 遍历行，并将对应行中的每个数据相乘
    # result_df = pd.DataFrame()
    # for index, row1 in df1.iterrows():
    #     row2 = df2.iloc[index]
    #     result_row = row1 * row2
    #     result_df = pd.concat([result_df, result_row], axis=1)
    #
    # # 转置结果DataFrame，使得行列对齐
    # result_df = result_df.T
    #
    # # 将结果保存为新的CSV文件
    # result_file_path = 'path/to/your/result_file.csv'
    # result_df.to_csv(result_file_path, index=False)
    #
    # print(f"结果已保存至: {result_file_path}")


def df_csv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))
    # 深度特征的key
    key = ['feature_{}'.format(i) for i in range(32000)]
    key.insert(0, 'label')
    df_val = pd.DataFrame(columns=key)
    df = pd.DataFrame(columns=key)
    # 创建模型实例
    model = mask_net()
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    # 指定.pth文件的路径
    file_path = 'log/cnn3_1220119_17_2020/epoch_2_CNNAt_0.978.pth'

    # 加载模型参数
    pp = torch.load(file_path)

    model.load_state_dict(pp)
    # 设置模型为评估模式
    model.eval()

    # 可以开始使用模型进行推断了
    path_dataset = "../bin/brats_npy/brats2020_3D_160"
    path_trainval_txt = ""
    with open(os.path.join(path_trainval_txt, "name_txt/train_name.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(path_trainval_txt, "name_txt/val_name.txt"), "r") as f:
        val_lines = f.readlines()
    train_dataset = BraTs2020(path_dataset, train_lines)
    train_num = len(train_dataset)
    print(train_num)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=1, num_workers=2,
                              pin_memory=True, drop_last=False)
    val_dataset = BraTs2020(path_dataset, val_lines)
    val_num = len(val_dataset)
    print(val_num)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=2,
                            pin_memory=True,
                            drop_last=False)
    with torch.no_grad():

        # 获得训练的深度特征
        for step, data in enumerate(train_loader, start=0):
            images, p_labels, labels = data
            logits, x = model(images.to(device))
            # 将fc的1000层转成list
            y_list = labels.tolist()
            X_list = x.tolist()
            temp = X_list[0]
            temp.insert(0, y_list[0])
            df.loc[step] = temp

        csv_train_path = './DF/train.csv'
        df.to_csv(csv_train_path, index=False)

        # 获得验证的深度特征
        for index_val, val_data in enumerate(val_loader):

            val_images, val_p_labels, val_labels = val_data
            outputs, x = model(val_images.to(device))
            print(val_labels)
            # 将验证集的转成df
            y_list = val_labels.tolist()
            X_list = x.tolist()
            temp = X_list[0]
            temp.insert(0, y_list[0])
            # 写入df
            df_val.loc[index_val] = temp
        csv_val_path = './DF/val.csv'
        df_val.to_csv(csv_val_path, index=False)
def get_scores():
    df = pd.read_csv('./DF/train_mask.csv')
    df_val = pd.read_csv('./DF/val_mask.csv')
    X_train = df[df.columns[1:]]
    # print(X_train)
    y_train = df['label']
    X_val = df_val[df_val.columns[1:]]
    y_val = df_val['label']
    # # svm
    # param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.1, 1]}
    # clf = svm.SVC()
    # grid_search = GridSearchCV(clf, param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    # # 输出最佳参数
    # print("最佳参数:", grid_search.best_params_)
    #
    # # 在测试集上评估模型
    # best_model = grid_search.best_estimator_
    # accuracy = best_model.score(X_val, y_val)
    # print("在测试集上的准确率:", accuracy)
    # rf
    best_acc = 0
    best_acc_1 =0
    for i in tqdm(range(100)):
        results_rf = []
        y_val_r = []
        true_num = 0
        rf = RandomForestClassifier(n_estimators=500, random_state=i)
        rf.fit(X_train, y_train)
        probabilities = rf.predict_proba(X_val)
        for index_result in tqdm(range(75)):
            results_rf.append(float(np.argmax(probabilities[3*index_result,:]+probabilities[3*index_result+1,:]+probabilities[3*index_result+2,:])))
            y_val_r.append((y_val.tolist()[3*index_result]+y_val.tolist()[3*index_result+1]+y_val.tolist()[3*index_result+2])/3)
        for j in range(len(results_rf)):
            if y_val_r[j]==results_rf[j]:
                true_num+=1
        acc_rf_1 = true_num/75
        best_acc_RF = rf.score(X_val, y_val)
        # print('acc_rf_1:', acc_rf_1)
        # print('best_acc_RF', best_acc_RF)
        if acc_rf_1>best_acc_1:
            best_acc_1 = acc_rf_1
            print( 'i:{}'.format(i),'rf_1:',acc_rf_1)
        if best_acc_RF>best_acc:
            best_acc = best_acc_RF
            print( 'i:{}'.format(i),'rf:',best_acc_RF)
def get_mask():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))
    # 可以开始使用模型进行推断了
    path_dataset = "../bin/brats_npy/brats2020_3D_160"
    path_trainval_txt = ""
    with open(os.path.join(path_trainval_txt, "name_txt/all_ids.txt"), "r") as f:
        mask_lines = f.readlines()
    mask_dataset = BraTs2020_mask(path_dataset, mask_lines)
    mask_num = len(mask_dataset)
    mask_loader = DataLoader(mask_dataset, shuffle=False, batch_size=1, num_workers=1,
                            pin_memory=True,
                            drop_last=False)

    # 深度特征的key
    key = ['feature_{}'.format(i) for i in range(32000)]
    df = pd.DataFrame(columns=key)
    # 创建模型实例
    model = mask_net()
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    def initialize_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    # 对模型进行权重初始化
    model.apply(initialize_weights)
    # 打印初始化后的权重
    for name, param in model.named_parameters():
        print(name, param)
    with torch.no_grad():
        for step, data in enumerate(mask_loader, start=0):
            images = data
            logits, x = model(images.to(device))
            x = x/67108864
            # 将fc的1000层转成list
            X_list = x.tolist()
            temp = X_list[0]
            df.loc[step] = temp
        csv_train_path = 'DF/mask_ori_cnn2.csv'
        df.to_csv(csv_train_path, index=False)



def rf_cl():
    df = pd.read_csv('./DF/train_mask.csv')
    df_val = pd.read_csv('./DF/val_mask.csv')
    X_train = df[df.columns[1:]]
    # print(X_train)
    y_train = df['label']
    X_val = df_val[df_val.columns[1:]]
    y_val = df_val['label']

    def self_paced_learning(X, y, max_iterations=10, initial_threshold=0.5, learning_rate=0.1):
        """
        自主学习算法，逐步调整样本权重
        """
        n_samples = len(y)
        sample_weights = np.ones(n_samples)  # 初始样本权重为1

        for iteration in tqdm(range(max_iterations)):
            # 创建随机森林分类器
            clf = RandomForestClassifier(n_estimators=50, random_state=iteration)

            # 使用带权重的样本进行训练
            clf.fit(X, y, sample_weight=sample_weights)

            # 预测训练数据
            y_pred = clf.predict(X)

            # 计算错误率
            error_rate = 1 - accuracy_score(y, y_pred)

            # 计算每个样本的误差
            individual_errors = np.abs(y_pred - y)

            # 计算下一轮的样本权重
            sample_weights *= np.exp(learning_rate * individual_errors)

            # 归一化样本权重
            sample_weights /= np.sum(sample_weights)

        return clf



    # 进行自主学习的随机森林训练
    self_paced_rf = self_paced_learning(X_train, y_train)

    # 在测试集上进行预测
    y_pred_test = self_paced_rf.predict(X_val)

    # 计算准确率
    accuracy = accuracy_score(y_val, y_pred_test)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    get_scores()