import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
# 1.读取训练数据集
data = pd.read_csv(r"bin/csv_data/t2-re_mask-1-test.csv")
X = data[data.columns[1:]]
Y = data["label"]
print(X.shape)

# 1.标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X)

# 2.构建RF模型
RFC_ = RFC()                               # 随机森林
c = RFC_.fit(X, Y).feature_importances_    # 特征重要性
print("重要性：")
print(c)

# 3. 交叉验证递归特征消除法
selector = RFECV(RFC_, step=1, cv=10)       # 采用交叉验证，每次排除一个特征，筛选出最优特征
selector = selector.fit(X, Y)
X_wrapper = selector.transform(X)          # 最优特征
score =cross_val_score(RFC_ , X_wrapper, Y, cv=10).mean()   # 最优特征分类结果
print(score)
print("最佳数量和排序")
print(selector.support_)                                    # 选取结果
print(selector.n_features_)                                 # 选取特征数量
print(selector.ranking_)                                    # 依次排数特征排序


# 4.递归特征消除法
selector1 = RFE(RFC_, n_features_to_select=3, step=1).fit(X, Y)       # n_features_to_select表示筛选最终特征数量，step表示每次排除一个特征
selector1.support_.sum()
print(selector1.ranking_)                                             # 特征排除排序
print(selector1.n_features_)                                          # 选择特征数量
X_wrapper1 = selector1.transform(X)                                   # 最优特征
score =cross_val_score(RFC_, X_wrapper1, Y, cv=9).mean()
print(score)

# 5.递归特征消除法和曲线图选取最优特征数量
score = []                                                            # 建立列表
for i in range(1, 8, 1):
    X_wrapper = RFE(RFC_, n_features_to_select=i, step=1).fit_transform(X, Y)    # 最优特征
    once = cross_val_score(RFC_, X_wrapper, Y, cv=9).mean()                      # 交叉验证
    score.append(once)                                                           # 交叉验证结果保存到列表
print(max(score), (score.index(max(score))*1)+1)                                 # 输出最优分类结果和对应的特征数量
print(score)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 8, 1), score)
plt.xticks(range(1, 8, 1))
plt.show()








