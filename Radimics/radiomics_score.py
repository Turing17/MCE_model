from feature_select import feature_selection
import pandas as pd
import os
import pickle
import numpy as np
def get_random_forest_image():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import tree

    # 生成随机数据
    np.random.seed(0)
    n_samples = 100
    X = np.random.rand(n_samples, 2) * 4 - 2
    y = np.array([0] * 50 + [1] * 50)

    # 创建随机森林分类器
    clf = RandomForestClassifier(n_estimators=1)

    # 使用数据集训练随机森林模型
    clf.fit(X, y)

    # 绘制每棵树
    n = 0
    for estimator in clf.estimators_:
        # 创建一个子图
        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制决策树
        tree.plot_tree(estimator, filled=True)

        # 设置标题
        ax.set_title('RandomForest')

        # 显示图像
        plt.show()

        n += 1

def get_txt_radiomics_socre():
    path_nametxt = '../bin/nametxt/brats'
    path_train_txt = os.path.join(path_nametxt, 'train_name.txt')
    path_val_txt = os.path.join(path_nametxt, 'val_name.txt')
    path_csv = './rf_csv/csv_with_label/t2_14.csv'
    with open(path_train_txt, 'r') as f:
        lines_train = f.readlines()
    with open(path_val_txt, 'r') as f:
        lines_val = f.readlines()
    list_train = []
    list_val = []
    # 获得样本划分list
    for line_train in lines_train:
        list_train.append(line_train.split()[0])
    for line_val in lines_val:
        list_val.append(line_val.split()[0])
    # # 根据样本划分list提取对应的train和val的dataframe
    data = pd.read_csv(path_csv)
    data_train = data[data['id'].isin(list_train)]
    data_val = data[data['id'].isin(list_val)]


    # 加载模型
    with open('../log/random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    # 使用加载的模型进行预测
    path_score_save = '../log/radiomics_score.txt'
    file_radiomics = open(path_score_save,'w+')

    x_val = data_val[data_val.columns[1:]]
    y_val = data_val['label']
    for index, row in x_val.iterrows():
        _, d = index, row
        d_val = d[1:].to_frame().T
        out = loaded_model.predict_proba(d_val)
        arr_str = np.array_str(out).strip('[]').replace('\n', '').replace('  ', ' ')
        id  = d[0]
        file_radiomics.write(id+" "+arr_str+' '+str(np.argmax(out)))
        file_radiomics.write('\n')
        print(id,arr_str)
    file_radiomics.close()
    # print()
def get_val_acc(p):
    print(p)
    path_nametxt = '../bin/nametxt/brats'
    path_train_txt = os.path.join(path_nametxt, 'train_name.txt')
    path_val_txt = os.path.join(path_nametxt, 'val_name.txt')
    path_csv = './rf_csv/csv_with_label/{}_14.csv'.format(p)
    with open(path_train_txt, 'r') as f:
        lines_train = f.readlines()
    with open(path_val_txt, 'r') as f:
        lines_val = f.readlines()
    list_train = []
    list_val = []
    # 获得样本划分list
    for line_train in lines_train:
        list_train.append(line_train.split()[0])
    for line_val in lines_val:
        list_val.append(line_val.split()[0])
    # # 根据样本划分list提取对应的train和val的dataframe
    data = pd.read_csv(path_csv)
    data_train = data[data['id'].isin(list_train)]
    data_val = data[data['id'].isin(list_val)]
    fs = feature_selection.feature_select(data_train)
    result = fs.RandomForest(data_val)
if __name__ == '__main__':
    # P = ["flair", "t1ce", "t1", "t2"]
    # for p in P:
    #     get_val_acc(p)
    get_random_forest_image()


