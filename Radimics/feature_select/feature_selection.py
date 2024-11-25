from scipy.stats import levene, ttest_ind
import pandas as pd
from sklearn.linear_model import LassoCV
from numpy import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif as mic
from sklearn.feature_selection import f_classif, f_regression, RFE

from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
import pickle
# 传入的数据为DataFrame,
class feature_select(object):
    def __init__(self, data):
        '''
        :param data: dataframe,label在第一列
        '''
        self.data = data
        # self.data_y = data_y
    def RandomForest(self,data_val):
        x_train = self.data[self.data.columns[2:]]
        y_train = self.data['label']
        x_val = data_val[data_val.columns[2:]]
        y_val = data_val['label']
        # 使用随机森林模型进行特征选择
        rf = RandomForestClassifier(n_estimators=10000, random_state=7)
        rf.fit(x_train, y_train)
        with open('../log/random_forest_model.pkl', 'wb') as file:
            pickle.dump(rf, file)
        print('Accuracy on training set: {:.3f}'.format(rf.score(x_train, y_train)))
        out = rf.predict_proba(x_val)
        print('Accuracy on test set: {:.3f}'.format(rf.score(x_val, y_val)))
        feature_scores=rf.feature_importances_
        # 输出特征重要性得分
        print("Feature importance scores:", feature_scores)
        # 选择特征重要性得分排名前2的特征
        feature_names = np.array(x_train.columns)  # 将 feature_names 转换为 numpy 数组
        selected_features = feature_names[np.argsort(rf.feature_importances_)[-10:]]
        return selected_features
    def enet(self):
        x = self.data[self.data.columns[2:]]
        y = self.data['label']
        # 选出最佳的alpha和l1_ratio
        model = ElasticNetCV(l1_ratio=0.5,
    eps=0.001,
    n_alphas=100,
    alphas=None,
    fit_intercept=True,
    normalize=True,
    precompute=False,
    max_iter=100000,
    tol=0.0000001,
    cv=10,
    copy_X=True,
    verbose=0,
    n_jobs=-1,
    positive=False,
    random_state=0,
    selection='cyclic')
        model.fit(x, y)
        print("Best alpha:", model.alpha_)
        print("Best l1_ratio:", model.l1_ratio_)
        # 使用选定的参数拟合ElasticNet模型
        elastic_net = ElasticNetCV(l1_ratio=model.l1_ratio_, cv=5, random_state=42)
        elastic_net.fit(x, y)
        # 获取每个特征的重要性得分
        feature_scores = np.abs(elastic_net.coef_)
        # feature_scores /= feature_scores.max()

        # 根据得分选择特征
        selected_features = np.where(feature_scores > 0.0)
        print("Selected features:", selected_features)
        return selected_features

    def t_test(self):
        '''
        本T检验只做独立检验，
        类别：2
        :return:
        counts:筛选的个数 int
        result_index:结果 list
        '''
        print("Start t-test:")
        row, _ = self.data.shape
        label_0 = self.data[self.data['label'] == 0]
        label_1 = self.data[self.data['label'] == 1]
        counts = 0
        result_index = []
        for colName in self.data.columns[2:]:
            # 检验是否具有方差齐性，p值大于0.05具有方差齐性
            if levene(label_0[colName], label_1[colName])[1] < 0.05:
                counts += 1
                result_index.append(colName)
            else:
                if ttest_ind(self.data[colName], self.data[colName], equal_var=False)[1] < 0.05:
                    counts += 1
                    result_index.append(colName)
        print("T-test picked " + str(counts) + " from " + str(_ - 1))

        return result_index

    def Lasso(self, alphas_min, alphas_max, alphas_num, cv, max_iter):
        '''
        :param alphas_min: 确定参数空间最小值
        :param alphas_max:
        :param alphas_num:
        :param cv: k折
        :param max_iter: 最大轮数
        :return:
        '''
        print('Start Lasso~~~')
        x = self.data[self.data.columns[2:]]
        y = self.data['label']
        row, _ = x.shape
        # alphas参数值，cv:交叉验证
        alphas = np.logspace(alphas_min, alphas_max, alphas_num)
        model_lassoCV = LassoCV(alphas=alphas,
                                cv=cv,
                                max_iter=max_iter).fit(x, y)
        print("End Lasso！")
        # 输出alpha的值
        print("Lasso_alpha：", model_lassoCV.alpha_)
        # 按照属性名，输出值
        coef = pd.Series(model_lassoCV.coef_, index=x.columns)
        print("Lasso picked " + str(sum(coef != 0)) + " from " + str(_))
        result_index = list(coef[coef != 0].index)
        x = x[result_index]

        return x, y

    def PCA(self, n_components):
        data = self.data
        x = data[data.columns[1:]]
        y = data["label"]
        pca = PCA(n_components=n_components)  # 加载PCA算法，设置降维后主成分数目为2
        x = pca.fit_transform(x)  # 对样本进行降维，data_pca降维后的数据
        x = pd.DataFrame(x)
        return x, y

    def F_test(self, k=10):
        data = self.data
        x = data[data.columns[1:]]
        y = data["label"]
        # k :选出的特征数
        model_F = SelectKBest(f_classif, k=k)
        model_F.fit(x, y)  # 分类问题
        index_feature = model_F.get_feature_names_out()
        x = x[index_feature]

        return x, y

    def Chi_test(self):
        pass

    def MI(self):
        data = self.data
        x = data[data.columns[1:]]
        y = data["label"]
        result = mic(x, y)
        k = result.shape[0] - sum(result <= 0)
        x_fsmic = SelectKBest(mic, k=k).fit(x, y)
        index_feature = x_fsmic.get_feature_names_out()
        print(index_feature.shape)
        print(index_feature)
        return result

    def rfe(self):
        print("Start rfe:")
        data = self.data
        x = data[data.columns[1:]]
        y = data["label"]
        # use linear regression as the model
        lr = LinearRegression()
        # rank all features, i.e continue the elimination until the last one
        rfe = RFE(lr, n_features_to_select=10)
        rfe.fit(x, y)
        index_feature = rfe.get_feature_names_out()
        print(index_feature)

    # def cox(self):
    #     data = self.data
    #     cph = CoxPHFitter()
    #     cph.fit(data, duration_col="label")
    #     cph.print_summary()
