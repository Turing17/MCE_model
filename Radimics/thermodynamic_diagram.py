# cmap(颜色)

import matplotlib.pyplot as plt
import seaborn as sns
# % matplotlib inline
df = 1
f, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=2)

# cmap用cubehelix map颜色
cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
pt = df.corr()  # pt为数据框或者是协方差矩阵
sns.heatmap(pt, linewidths=0.05, ax=ax1, vmax=900, vmin=0, cmap=cmap)
ax1.set_title('cubehelix map')
ax1.set_xlabel('')
ax1.set_xticklabels([])  # 设置x轴图例为空值
ax1.set_ylabel('kind')

# cmap用matplotlib colormap
sns.heatmap(pt, linewidths=0.05, ax=ax2, vmax=900, vmin=0, cmap='rainbow')
# rainbow为 matplotlib 的colormap名称
ax2.set_title('matplotlib colormap')
ax2.set_xlabel('region')
ax2.set_ylabel('kind')
# center的用法(颜色)

f, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=2)

cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
sns.heatmap(pt, linewidths=0.05, ax=ax1, cmap=cmap, center=None)
ax1.set_title('center=None')
ax1.set_xlabel('')
ax1.set_xticklabels([])  # 设置x轴图例为空值
ax1.set_ylabel('kind')

# 当center设置小于数据的均值时，生成的图片颜色要向0值代表的颜色一段偏移
sns.heatmap(pt, linewidths=0.05, ax=ax2, cmap=cmap, center=200)
ax2.set_title('center=3000')
ax2.set_xlabel('region')
ax2.set_ylabel('kind')