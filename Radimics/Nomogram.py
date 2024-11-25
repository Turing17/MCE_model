# 绘制列线图

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib


# 注意注释掉此句，才可以plt.show()看到效果图
# 设置此才可保存图片到本地，plt.savefig将保存结果图，但同时plt.show()将不在起作用
# matplotlib.use("Agg")

# 设置展示的刻度
# 设置刻度轴位置
# 刻度起始值、结束值、刻度最小精度、刻度间隔
# 文字位置
def setup(ax, title, minx, maxx, major, minor, deviation=-1, position="bottom"):
    # 只显示底部脊椎
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    if (position == "bottom"):
        ax.spines['top'].set_color('none')
    elif (position == "top"):
        ax.spines['bottom'].set_color('none')

    # 定义刻度最大最小精度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor))  # 最小刻度精度
    # ax.xaxis.set_ticks_position()

    # 定义刻度位置
    ax.xaxis.set_ticks_position(position)
    # ax.xaxis(fontsize=10)
    ax.set_xlim(minx, maxx)
    ax.text(-0.5, -1, title, transform=ax.transAxes,
            fontsize=9, fontname='Monospace', color='black')


# plt.figure(figsize=(32, 8))
fig, axs = plt.subplots(6, 1, figsize=(10, 6))
# fig.suptitle("Nomogram demo") # 设置标题

setup(axs[0], title="Points", position="top", minx=0, maxx=100, major=10, minor=2.5,deviation=-1)
axs[0].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

setup(axs[1], title="t test^p:", minx=0.05, maxx=0, major=0.05, minor=1,deviation=-1)
# axs[1].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

setup(axs[2], title="CCC:", minx=0, maxx=1, major=0.1, minor=0.1,deviation=-1)
# axs[2].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

setup(axs[3], title="Information gain", minx=0, maxx=10, major=1, minor=0.1,deviation=-1)
# axs[3].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

setup(axs[4], title="TotalPoints", minx=0, maxx=260, major=20, minor=4,deviation=5)
# axs[4].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# setup(axs[5], title="Linear Predictor", minx=-0.8, maxx=0.8, major=0.2, minor=0.1,deviation=2)
# axs[5].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

setup(axs[5], title="result", minx=0, maxx=1, major=1, minor=1,deviation=3)
# axs[6].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#
# setup(axs[7], title="3-yearSurvival Probability", minx=0.5, maxx=0.05, major=0.05, minor=0.05)
# axs[7].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#
# setup(axs[8], title="5-yearSurvival Probability", minx=0.25, maxx=0.05, major=0.05, minor=0.05)
# axs[8].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
fig.tight_layout()
plt.show()

# 保存图片
# plt.savefig('nomogram.jpg')
