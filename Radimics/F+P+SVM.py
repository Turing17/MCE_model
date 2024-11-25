import pandas as pd
from numpy import *
from utils_.data_prepare import data_pre
from classifier import classifier_x
import joblib
import time
import os
from Radiomics.feature_select.feature_selection import feature_select

data_path = "bin/csv_data/flair-re_mask-1-test1.csv"
name_path = "./bin/feature_name/name_txt/"
model_path = "./bin/"
if __name__ == '__main__':
    # 导入数据
    data = pd.read_csv(data_path)
    data = data_pre(data).prepare()
    train_txt = name_path + "train.txt"
    file = open(train_txt, "r")
    train_list = []
    for i in file.read().split("\n")[:-2]:
        train_list.append(int(i))
    data = data.iloc[train_list]
    # x = data[data.columns[1:]]
    # y = data["label"]
    # 导入结束
    # 特征选择
    x, y = feature_select(data).F_test(k=10)
    index_select = list(x.columns)
    # print(index_lasso)
    # 记录选择的特征名字
    file = open('./bin/feature_name/file_name.txt', 'w')
    for i in index_select:
        file.writelines(str(i) + "\n")
    file.close()
    # 特征选择

    # 创建模型存储路径

    result_score = []
    now = time.strftime("%Y-%m-%d")
    model_save = os.path.join(model_path, "model_", now)
    folder = os.path.exists(model_save)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(model_save)  # makedirs 创建文件时如果路径不存在会创建这个路径
    C, gamma = classifier_x(x, y).svm_GridSearchCV()
    for i in range(10):
        # now = time.strftime("%Y-%m-%d-%H:%M")
        print('Epoch：{}/100'.format(i))

        model, score = classifier_x(x, y).K_fold_cross_validation(C, gamma)
        result_score.append(mean(score))
        # os.makedirs("./bin/model/model{}.pickle".format(i))
        file = "{}/model_{}_{}.m".format(model_save, i, float(mean(score)), ".2f")
        joblib.dump(model, file)

    print('The best result:', max(result_score))
    print("The worst result:", min(result_score))
    print('The mean:', mean(result_score))
