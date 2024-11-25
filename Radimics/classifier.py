from Radiomics.feature_select.feature_selection import feature_select
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import svm
from numpy import *


class classifier_x(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def RF(self, tree_n, x_y_p):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=x_y_p)
        # n_estimators=20，20棵树
        model_rf = RandomForestClassifier(n_estimators=tree_n).fit(x_train, y_train)
        score_rf = model_rf.score(x_test, y_test)
        print('rf_score:', score_rf)
        return float(score_rf)

    def SVM(self, x_y_p, C, gamma, kernel="rbf"):
        # 优化,base=2，以2为底
        # SVM
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=x_y_p)
        model_svm = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True).fit(x_train, y_train)
        score_svm = model_svm.score(x_test, y_test)
        print('score_svm:', score_svm)
        return model_svm, score_svm

    def svm_GridSearchCV(self):
        # 优化
        # 优化,base=2，以2为底
        Cs = np.logspace(-2, 4, 50, base=2)
        gammas = np.logspace(-5, 2, 100, base=2)
        param_grid = dict(C=Cs, gamma=gammas)
        grid = GridSearchCV(svm.SVC(kernel="rbf"), param_grid=param_grid, cv=10).fit(self.x, self.y)
        print()
        print("最优参数", grid.best_params_)
        C = grid.best_params_["C"]
        gamma = grid.best_params_["gamma"]
        return C, gamma

    def K_fold_cross_validation(self, C, gamma):
        '''
        K折P次交叉验证
        n_splists:K折
        n_repeats:P次
        :return:
        '''

        x = self.x
        y = self.y
        result = []
        rkf = RepeatedKFold(n_splits=3, n_repeats=2)
        for train_index, test_index in rkf.split(x):
            x_train = x.iloc[train_index]
            x_test = x.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            model_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True).fit(x_train, y_train)
            score_svm = model_svm.score(x_test, y_test)
            result.append(float(score_svm))
            # print('*' * 50)
            # print(score_svm)
            # print('*' * 50)
        return model_svm, result

    def logistic_regression(self, x_y_p):
        x = self.x
        y = self.y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=x_y_p)
        log_model = LogisticRegression().fit(x_train, y_train)
        log_score = log_model.score(x_test, y_test)
        return log_model, log_score


if __name__ == '__main__':
    csv1_filePath = r"./bin/result_f/im_lab_test.csv"


    def t_L_s_k(csv1_filePath):
        data_1 = pd.read_csv(csv1_filePath)
        # print(type(data_1))
        num, index = feature_select(data_1).t_test()
        index.insert(0, 'label')
        df = data_1[index]
        result_1 = []
        result, x, y = feature_select(df).Lasso(alphas_min=-3, alphas_max=1, alphas_num=50, cv=10, max_iter=100000)
        for i in range(100):
            print('Epoch：{}/100'.format(i))
            result_1.append(mean(classifier_x(x, y).K_fold_cross_validation()))
        print('The best result:', max(result_1))
        print("The worst result:", min(result_1))
        print('The mean:', mean(result_1))


    def l_s(path):
        data = pd.read_csv(path)
        result_1 = []
        index, x, y = feature_select(data).Lasso(-3, 1, 50, 10, 100000)
        for i in range(100):
            print('Epoch：{}/100'.format(i))
            result_1.append(mean(classifier_x(x, y).K_fold_cross_validation()))
        print('The best result:', max(result_1))
        print("The worst result:", min(result_1))
        print('The mean:', mean(result_1))
        # print(index)


    l_s(csv1_filePath)
