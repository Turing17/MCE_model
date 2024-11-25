import pandas as pd
from numpy import *
from utils_.data_prepare import data_pre
from classifier import classifier_x
import joblib
import time
import os
from Radiomics.feature_select.feature_selection import feature_select
data_path = "./bin/result_f/im_lab_test.csv"
name_path = "./bin/feature_name/name_txt/"
model_path = "./bin/"
if __name__ == '__main__':
    # 导入数据
    data = pd.read_csv(data_path)
    data = data_pre(data).prepare()
    train_txt = name_path+"train.txt"
    file = open(train_txt, "r")
    train_list = []
    for i in file.read().split("\n")[:-2]:
        train_list.append(int(i))
    data = data.iloc[train_list]
    # x = data[data.columns[1:]]
    # y = data["label"]
    result, x, y = feature_select(data).Lasso(alphas_min=-3,
                                                alphas_max=1,
                                                alphas_num=50,
                                                cv=10,
                                                max_iter=100000)
    # 导入结束
    index_lasso = list(x.columns)
    # print(index_lasso)
    file = open('./bin/feature_name/file_name.txt', 'w')
    for i in index_lasso:
        file.writelines(str(i) + "\n")
    file.close()

    result_score = []
    now = time.strftime("%Y-%m-%d")
    print(now)
    model_save = os.path.join(model_path, "model_", now)
    print(model_save)
    os.makedirs(model_save)
    for i in range(10):
        # now = time.strftime("%Y-%m-%d-%H:%M")
        print('Epoch：{}/100'.format(i))
        model, score = classifier_x(x, y).logistic_regression(x_y_p=0.3)
        result_score.append(mean(score))
        # os.makedirs("./bin/model/model{}.pickle".format(i))
        file = "{}/model_{}_{}.m".format(model_save,i, float(mean(score)), ".2f")
        joblib.dump(model, file)

    print('The best result:', max(result_score))
    print("The worst result:", min(result_score))
    print('The mean:', mean(result_score))
