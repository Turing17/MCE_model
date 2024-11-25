import numpy as np
import matplotlib.pyplot as plt

# 生成随机分类数据
# np.random.seed(100)
X1 = np.random.randn(288, 2) + [1, 1]
X2 = np.random.randn(77, 2) + [-1, -1]
X = np.vstack((X1, X2))
y = np.array([1] * 288 + [-1] * 77)
from sklearn.svm import SVC

# 训练SVM模型
# ["linear", "poly", "rbf", "sigmoid", "precomputed"]
svm = SVC(kernel='sigmoid')
svm.fit(X, y)

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel( ())
plt.ylabel( ())
plt.title('SVM')
plt.show()