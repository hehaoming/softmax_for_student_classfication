# coding=utf-8
"""通过动画展示最终结果
@author: hehaoming
@time: 2019/10/29 21:00
"""
import numpy as np
import logging

logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S",
                    level=logging.DEBUG)
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import read_data as rd

# train_data传入你处理后的数据
def show_binary_result(train_data, train_labels, error_list, theta_list, iterator):
    """迭代次数iterator从0（即对应初始状态的参数和损失）开始计数， error_list和theta_list下标从0开始"""
    # 数据预处理
    train_data_zeros = train_data[train_labels == 0]
    train_data_ones = train_data[train_labels == 1]
    error = []
    error_x = []

    # 画布
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=4)

    # 直线分割图
    ax[0].set_xlim(min(train_data[:, 1]) * 1.1, max(train_data[:, 1]) * 1.1)
    ax[0].set_ylim(min(train_data[:, 2]) * 1.1, max(train_data[:, 2]) * 1.1)
    # ax[0].set_yticks(np.linspace(40, 90, 11))
    # ax[0].set_xticks(np.linspace(15, 65, 11))
    ax[0].set_title("Classify")
    ax[0].set_xlabel("Exam 1 Score")
    ax[0].set_ylabel("Exam 2 Score")
    ax[0].plot(train_data_zeros[:, 1], train_data_zeros[:, 2], 'bo', markersize=2)
    ax[0].plot(train_data_ones[:, 1], train_data_ones[:, 2], 'r+', markersize=3)
    x = np.linspace(min(train_data[:, 1]), max(train_data[:, 1]), 11)
    logging.debug(x)
    line, = ax[0].plot(x, [np.nan] * len(x))

    # 误差变化图
    ax[1].set_title("Error Variety")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Mean Error")
    ax[1].set_ylim(0, max(error_list) * 1.1)
    ax[1].set_xlim(0, 10)
    error_line, = ax[1].plot([], [], 'ro', markersize=2)

    def init():
        line.set_ydata([np.nan] * len(x))
        error_line.set_ydata([np.nan] * len(error_x))
        return error_line, line

    def animate(i):
        # 更新error_line
        xmin, xmax = ax[1].get_xlim()
        if i >= xmax - 5:
            ax[1].set_xlim(int(xmin), xmax + 10)
            ax[1].set_xticks(np.linspace(int(xmin), int(xmax) + 10, 11))
            ax[1].figure.canvas.draw()
        error.append(error_list[i])
        error_x.append(i)
        error_line.set_data(error_x, error)

        # 更新分割直线
        y_data = f(theta_list[i][0], theta_list[i][1], theta_list[i][2], x)
        logging.debug("y_data: %s", str(y_data))
        line.set_ydata(y_data)
        return error_line, line

    def f(theta0, theta1, theta2, x1):
        return -(theta0 / theta2) - (theta1 / theta2) * x1

    ani = animation.FuncAnimation(
        fig, animate, frames=iterator - 1, init_func=init, interval=1000, blit=False, repeat=False)
    plt.show()


def show_multi_result(train_data, train_labels, error_list, theta_list, iterator):
    """
    迭代次数iterator从0（即对应初始状态的参数和损失）开始计数， error_list为一维向量，theta为三维矩阵
    """
    # 数据预处理
    train_data_zeros = train_data[train_labels == 0]
    train_data_ones = train_data[train_labels == 1]
    train_data_twos = train_data[train_labels == 2]
    error = []
    error_x = []

    # 画布
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=4)

    # 直线分割图
    ax[0].set_xlim(min(train_data[:, 1]) * 1.1, max(train_data[:, 1]) * 1.1)
    ax[0].set_ylim(min(train_data[:, 2]) * 1.1, max(train_data[:, 2]) * 1.1)
    ax[0].set_title("Classify")
    ax[0].set_xlabel("feature 1")
    ax[0].set_ylabel("feature 2 ")
    ax[0].plot(train_data_zeros[:, 1], train_data_zeros[:, 2], 'bo', markersize=2)
    ax[0].plot(train_data_ones[:, 1], train_data_ones[:, 2], 'r+', markersize=3)
    ax[0].plot(train_data_twos[:, 1], train_data_twos[:, 2], 'g^', markersize=3)

    x = np.linspace(min(train_data[:, 1]), max(train_data[:, 1]), 11)
    print(x)
    line1, = ax[0].plot(x, [np.nan] * len(x))
    line2, = ax[0].plot(x, [np.nan] * len(x))
    line3, = ax[0].plot(x, [np.nan] * len(x))

    # 误差变化图
    ax[1].set_title("Error Variety")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Mean Error")
    ax[1].set_ylim(0, max(error_list) * 1.1)
    ax[1].set_xlim(0, 10)
    error_line, = ax[1].plot([], [], 'ro', markersize=2)

    def init():
        line1.set_ydata([np.nan] * len(x))
        line2.set_ydata([np.nan] * len(x))
        line3.set_ydata([np.nan] * len(x))
        error_line.set_ydata([np.nan] * len(error_x))
        return error_line, line1, line2, line3

    def animate(i):
        # 更新error_line
        xmin, xmax = ax[1].get_xlim()
        if i >= xmax - 5:
            ax[1].set_xlim(int(xmin), xmax + 10)
            ax[1].set_xticks(np.linspace(int(xmin), int(xmax) + 10, 11))
            ax[1].figure.canvas.draw()
        error.append(error_list[i])
        error_x.append(i)
        error_line.set_data(error_x, error)

        # 更新分割直线
        y_data1 = f(theta_list[i][0][0], theta_list[i][0][1], theta_list[i][0][2], x)
        logging.debug("y_data1: %s", str(y_data1))
        line1.set_ydata(y_data1)
        y_data2 = f(theta_list[i][1][0], theta_list[i][1][1], theta_list[i][1][2], x)
        logging.debug("y_data2: %s", str(y_data2))
        line2.set_ydata(y_data2)
        y_data3 = f(theta_list[i][1][0], theta_list[i][1][1], theta_list[i][1][2], x)
        logging.debug("y_data3: %s", str(y_data3))
        line3.set_ydata(y_data3)
        return error_line, line1, line2, line3

    def f(theta0, theta1, theta2, x1):
        return -(theta0 / theta2) - (theta1 / theta2) * x1

    ani = animation.FuncAnimation(
        fig, animate, frames=iterator - 1, init_func=init, interval=1000, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    show_multi_result(rd.read_data_from_resource("dataset2")[0],
                      rd.read_data_from_resource("dataset2")[1],
                      [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                      [[[-1 / 7, -2, 4.5],
                        [-0, -1, 2],
                        [-0.2, 0.5, 5]],
                       [[-1, -1, 1.8],
                        [-0, -1.1, 2],
                        [-0.1, -0.5, 6]],
                       [[-1, -1, 1.7],
                        [-0, -1.2, 2],
                        [-0.1, -0.6, 7]]
                       ], 4)
