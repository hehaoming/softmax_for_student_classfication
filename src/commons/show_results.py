# coding=utf-8
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import read_data


def show_result(train_data, train_labels, error_list, theta_list, iterator):  # 迭代次数iterator从1开始计数
    # 数据预处理
    train_data_zeros = train_data[train_labels == 0]
    train_data_ones = train_data[train_labels == 1]
    error = []
    error_x = []

    # 画布
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=4)

    # 直线分割图
    ax[0].set_xlim(15 - 15 * 0.2, 65 + 65 * 0.2)
    ax[0].set_ylim(40 - 40 * 0.2, 90 + 90 * 0.2)
    ax[0].set_yticks(np.linspace(40, 90, 11))
    ax[0].set_xticks(np.linspace(15, 65, 11))
    ax[0].set_title("Classify")
    ax[0].set_xlabel("Exam 1 Score")
    ax[0].set_ylabel("Exam 2 Score")
    ax[0].plot(train_data_zeros[:, 1], train_data_zeros[:, 2], 'bo', markersize=2)
    ax[0].plot(train_data_ones[:, 1], train_data_ones[:, 2], 'r+', markersize=3)
    x = np.arange(15, 65, 1)
    print(x)
    line, = ax[0].plot(x, [np.nan] * len(x))

    # 误差变化图
    ax[1].set_title("Error Variety")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Mean Error")
    ax[1].set_ylim(0, error_list[0] * 1.2)
    ax[1].set_xlim(0, 10)
    error_line, = ax[1].plot([], [], 'ro', markersize=2)

    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(x))
        error_line.set_ydata([np.nan] * len(error_x))
        return error_line, line

    def animate(i):
        # 更新error_line
        error.append(error_list[i])
        error_x.append(i)
        error_line.set_data(error_x, error)

        # 更新分割直线
        line.set_ydata(f(theta_list[i][0], theta_list[i][1], theta_list[i][2], x))
        return error_line, line

    def f(theta0, theta1, theta2, x1):
        return (-theta0 / theta2) - (theta1 / theta2) * x1

    ani = animation.FuncAnimation(
        fig, animate, frames=iterator - 1, init_func=init, interval=1000, blit=False)
    plt.show()


# show_result(read_data.read_data_from_resource()[0],
#             read_data.read_data_from_resource()[1],
#             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
#             [[-190, 3, 2],
#              [-200, 4, 3],
#              [-190, 3, 2],
#              [-200, 4, 3],
#              [-190, 3, 2],
#              [-200, 4, 3],
#             ])