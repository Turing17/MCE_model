from utils_.data_prepare import data_pre
import pandas as pd
from Radiomics.feature_select.feature_selection import feature_select
from numpy import *
from classifier import classifier_x
import joblib

no_csv = "flair-re"
mask = 'mask-1'
no_lab = "{}_{}".format(no_csv,mask)
csv1_filePath = r"./bin/csv_data/{}-test1.csv".format(no_lab)
train_txt = r"./bin/feature_name/name_txt/train.txt"


def t_L_s_k(csv1_filePath, train_txt):
    file = open(train_txt, "r")
    train_list = []
    for i in file.read().split("\n")[:-2]:
        train_list.append(int(i))
    data_1 = pd.read_csv(csv1_filePath)
    # print(type(data_1))
    # num, index = feature_select(data_1).t_test()
    # df = data_1[index]
    data_1 = data_1.iloc[train_list]
    data = data_pre(data_1).prepare()
    # index_t_test = feature_select(data_1).t_test()
    # data = data_1[index_t_test]
    # data.insert(0,"label",data_1[data_1.columns[0]])



    result_1 = []
    x, y = feature_select(data).Lasso(alphas_min=-3,
                                        alphas_max=1,
                                        alphas_num=50,
                                        cv=10,
                                        max_iter=10000)
    index_lasso = list(x.columns)
    # print(index_lasso)
    file = open('./bin/result/{}/file_name1.txt'.format(no_lab), 'w')
    for i in index_lasso:
        file.writelines(str(i) + "\n")
    file.close()

    C, gamma = classifier_x(x, y).svm_GridSearchCV()

    for i in range(5):
        # now = time.strftime("%Y-%m-%d-%H:%M")
        print('Epoch：{}/100'.format(i))
        model, re_score = classifier_x(x, y).SVM(0.2,C, gamma)
        result_1.append(mean(re_score))
        # os.makedirs("./bin/model/model{}.pickle".format(i))
        file = "./bin/result/{}/model1/model_{}_{}.m".format(no_lab,i, float(mean(re_score)), ".2f")
        joblib.dump(model, file)

    print('The best result:', max(result_1))
    print("The worst result:", min(result_1))
    print('The mean:', mean(result_1))


if __name__ == '__main__':
    t_L_s_k(csv1_filePath, train_txt)
