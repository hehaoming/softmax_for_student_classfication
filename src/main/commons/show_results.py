# coding=utf-8
"""通过动画展示最终结果
@author: hehaoming
@time: 2019/10/29 21:00
"""
import numpy as np
import logging
import read_data as rd
from queue import Queue
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation


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
    min_train1 = min(train_data[:, 1])
    max_train1 = max(train_data[:, 1])
    min_train2 = min(train_data[:, 2])
    max_train2 = max(train_data[:, 2])
    ax[0].set_xlim(min_train1 - 0.1 * abs(min_train1), max_train1 + 0.1 * abs(max_train1))
    ax[0].set_ylim(min_train2 - 0.1 * abs(min_train2), max_train2 + 0.1 * abs(max_train2))
    # ax[0].set_yticks(np.linspace(40, 90, 11))
    # ax[0].set_xticks(np.linspace(15, 65, 11))
    ax[0].set_title("Classify")
    ax[0].set_xlabel("Exam 1 Score")
    ax[0].set_ylabel("Exam 2 Score")
    ax[0].plot(train_data_zeros[:, 1], train_data_zeros[:, 2], 'bo', markersize=2)
    ax[0].plot(train_data_ones[:, 1], train_data_ones[:, 2], 'r+', markersize=3)
    x = np.linspace(min_train1 - 0.2 * abs(min_train1), max_train1 + 0.2 * abs(max_train1), 11)
    logging.debug(x)
    line, = ax[0].plot(x, [np.nan] * len(x))

    # 误差变化图
    ax[1].set_title("Error Variety")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Mean Error")
    ax[1].set_ylim(min(error_list) * 0.9, max(error_list) * 1.1)
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
        line.set_data(*draw_line(i))
        return error_line, line

    def f(theta0, theta1, theta2, x1):
        return -(theta0 / theta2) - (theta1 / theta2) * x1

    def draw_line(frame):
        if theta_list[frame][2] != 0:
            y_data = f(theta_list[frame][0], theta_list[frame][1], theta_list[frame][2], x)
            x_data = x
        elif theta_list[frame][1] != 0:
            y_data = np.array([min(train_data[:, 2]), max(train_data[:, 2])])
            certain_value = - theta_list[frame][0] / theta_list[frame][1]
            x_data = np.array([certain_value, certain_value])
        else:
            x_data = x
            y_data = [np.nan] * len(x)
        logging.debug("y_data: %s %s", str(x_data), str(y_data))
        return x_data, y_data

    ani = animation.FuncAnimation(
        fig, animate, frames=int(iterator - 1), init_func=init, interval=0.1, blit=False, repeat=False)
    plt.show()


def show_softmax_binary_result(train_data, train_labels, error_list, theta_list, iterator):
    """
    迭代次数iterator从0（即对应初始状态的参数和损失）开始计数， error_list为一维向量，theta为三维矩阵
    num_class为类别数
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
    min_train1 = min(train_data[:, 1])
    max_train1 = max(train_data[:, 1])
    min_train2 = min(train_data[:, 2])
    max_train2 = max(train_data[:, 2])
    ax[0].set_xlim(min_train1 - 0.1 * abs(min_train1), max_train1 + 0.1 * abs(max_train1))
    ax[0].set_ylim(min_train2 - 0.1 * abs(min_train2), max_train2 + 0.1 * abs(max_train2))
    ax[0].set_title("Classify")
    ax[0].set_xlabel("feature 1")
    ax[0].set_ylabel("feature 2 ")
    ax[0].plot(train_data_zeros[:, 1], train_data_zeros[:, 2], 'bo', markersize=2)
    ax[0].plot(train_data_ones[:, 1], train_data_ones[:, 2], 'r+', markersize=3)
    ax[0].plot(train_data_twos[:, 1], train_data_twos[:, 2], 'g^', markersize=3)

    x = np.linspace(min_train1 - 0.2 * abs(min_train1), max_train1 + 0.2 * abs(max_train1), 11)
    print(x)
    line1, = ax[0].plot(x, [np.nan] * len(x))

    # 误差变化图
    ax[1].set_title("Error Variety")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Mean Error")
    ax[1].set_ylim(min(error_list) * 0.9, max(error_list) * 1.1)
    ax[1].set_xlim(0, 10)
    error_line, = ax[1].plot([], [], 'ro', markersize=2)

    def init():
        line1.set_ydata([np.nan] * len(x))
        error_line.set_ydata([np.nan] * len(error_x))
        return error_line, line1

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

        # 计算theta
        theta0 = theta_list[i][0][0] - theta_list[i][0][1]
        theta1 = theta_list[i][1][0] - theta_list[i][1][1]
        theta2 = theta_list[i][2][0] - theta_list[i][2][1]
        # 更新分割直线
        line1.set_data(*draw_line(theta0, theta1, theta2))
        return error_line, line1

    def f(x1, theta0, theta1, theta2):
        return -(theta0 / theta2) - (theta1 / theta2) * x1

    def draw_line(theta0, theta1, theta2):
        if theta2 != 0:
            y_data = f(x, theta0, theta1, theta2)
            x_data = x
        elif theta1 != 0:
            y_data = np.array([min(train_data[:, 2]), max(train_data[:, 2])])
            certain_value = - theta0 / theta1
            x_data = np.array([certain_value, certain_value])
        else:
            x_data = x
            y_data = [np.nan] * len(x)
        logging.debug("y_data: %s %s", str(x_data), str(y_data))
        return x_data, y_data

    ani = animation.FuncAnimation(
        fig, animate, frames=int(iterator - 1), init_func=init, interval=0.1, blit=False, repeat=False)
    plt.show()


def show_softmax_multi_result(train_data, train_labels, error_list, theta_list, iterator):
    """
    迭代次数iterator从0（即对应初始状态的参数和损失）开始计数， error_list为一维向量，theta为三维矩阵
    num_class为类别数
    """
    # 数据预处理
    train_data_zeros = train_data[train_labels == 0]
    train_data_ones = train_data[train_labels == 1]
    train_data_twos = train_data[train_labels == 2]
    error = []
    error_x = []

    # 预处理算好theta

    # 画布
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=4)

    # 直线分割图
    min_train1 = min(train_data[:, 1])
    max_train1 = max(train_data[:, 1])
    min_train2 = min(train_data[:, 2])
    max_train2 = max(train_data[:, 2])
    ax[0].set_xlim(min_train1 - 0.1 * abs(min_train1), max_train1 + 0.1 * abs(max_train1))
    ax[0].set_ylim(min_train2 - 0.1 * abs(min_train2), max_train2 + 0.1 * abs(max_train2))
    ax[0].set_title("Classify")
    ax[0].set_xlabel("feature 1")
    ax[0].set_ylabel("feature 2 ")
    ax[0].plot(train_data_zeros[:, 1], train_data_zeros[:, 2], 'bo', markersize=2)
    ax[0].plot(train_data_ones[:, 1], train_data_ones[:, 2], 'r+', markersize=3)
    ax[0].plot(train_data_twos[:, 1], train_data_twos[:, 2], 'g^', markersize=3)
    x = np.linspace(min_train1 - 0.2 * abs(min_train1), max_train1 + 0.2 * abs(max_train1), 101)

    fill_color = ['lightsteelblue', 'lightcoral', 'lightgreen']
    area_queue = Queue()
    for i in range(3):
        area_queue.put(ax[0].fill_between(x, [np.nan] * len(x), [np.nan] * len(x), color=fill_color[i], alpha=0.5))
    # 误差变化图
    ax[1].set_title("Error Variety")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Mean Error")
    ax[1].set_ylim(0, max(error_list) * 1.1)
    ax[1].set_xlim(0, 10)
    error_line, = ax[1].plot([], [], 'ro', markersize=1)

    def init():
        for k in range(3):
            a_area = area_queue.get()
            a_area.remove()
            area_queue.put(ax[0].fill_between(x, [np.nan] * len(x), [np.nan] * len(x), color=fill_color[k], alpha=0.5))
        error_line.set_ydata([np.nan] * len(error_x))
        return error_line, area_queue

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

        # 计算theta
        theta = []
        for k in range(3):
            area_line = []
            for j in range(1, 3, 1):
                theta0 = theta_list[i][0][k] - theta_list[i][0][(k + j) % 3]
                theta1 = theta_list[i][1][k] - theta_list[i][1][(k + j) % 3]
                theta2 = theta_list[i][2][k] - theta_list[i][2][(k + j) % 3]
                area_line.append([theta0, theta1, theta2])
            theta.append(area_line)
        for k in range(3):
            a_area = area_queue.get()
            a_area.remove()
            line0 = theta[k][0]
            line1 = theta[k][1]
            y_low = []
            y_top = []
            if line0[2] != 0. and line1[2] != 0.:
                path0 = f(x, *line0)
                path1 = f(x, *line1)
                if line0[2] > 0. and line1[2] > 0.:
                    for n in range(x.shape[0]):
                        if path0[n] > path1[n]:
                            y_low.append(path0[n])
                        else:
                            y_low.append(path1[n])
                        y_top.append(np.exp(10))
                elif line0[2] < 0. and line1[2] < 0.:
                    for n in range(x.shape[0]):
                        if path0[n] < path1[n]:
                            y_top.append(path0[n])
                        else:
                            y_top.append(path1[n])
                        y_low.append(-np.exp(10))
                elif line0[2] > 0. and line1[2] < 0.:
                    y_top = path1
                    y_low = path0
                elif line0[2] < 0. and line1[2] > 0.:
                    y_top = path0
                    y_low = path1
                area_queue.put(
                    ax[0].fill_between(x, y_low, y_top, where=(y_low < y_top), color=fill_color[k], alpha=0.5))
            elif line0[2] != 0. or line1[2] != 0.:
                if line0[2] != 0. and line1[2] == 0.:
                    path0 = f(x, *line0)
                    if line0[2] > 0:
                        y_low = path0
                        y_top = [np.exp(10)] * x.shape[0]
                    if line0[2] < 0:
                        y_low = [-np.exp(10)] * x.shape[0]
                        y_top = path0
                elif line0[2] == 0. and line1[2] != 0:
                    path1 = f(x, *line1)
                    if line1[2] > 0:
                        y_low = path1
                        y_top = [np.exp(10)] * x.shape[0]
                    if line1[2] < 0:
                        y_low = [-np.exp(10)] * x.shape[0]
                        y_top = path1
                new_y_top = []
                new_y_low = []
                new_x = []
                for m in range(len(x)):
                    if line1[1] * x[m] > -line1[0]:
                        new_x.append(x[m])
                        new_y_top.append(y_top[m])
                        new_y_low.append(y_low[m])
                area_queue.put(ax[0].fill_between(new_x, new_y_low, new_y_top, where=(new_y_low < new_y_top), color=fill_color[k], alpha=0.5))
            elif line0[2] == 0. and line1[2] == 0:
                y_low = [-np.exp(10)] * x.shape[0]
                y_top = [np.exp(10)] * x.shape[0]
                new_y_top = []
                new_y_low = []
                new_x = []
                for m in range(len(x)):
                    if line0[1] * x[m] + line0[0] > 0 and line1[1] * x[m] + line1[0] > 0:
                        new_x.append(x[m])
                        new_y_top.append(y_top[m])
                        new_y_low.append(y_low[m])
                area_queue.put(ax[0].fill_between(new_x, new_y_low, new_y_top, where=(new_y_low < new_y_top), color=fill_color[k], alpha=0.5))
        return error_line, area_queue

    def f(x1, theta0, theta1, theta2):
        return -(theta0 / theta2) - (theta1 / theta2) * x1

    ani = animation.FuncAnimation(
        fig, animate, frames=int(iterator - 1), init_func=init, interval=0.1, blit=False, repeat=False)
    plt.show()


def test_show_multi_result():
    show_softmax_multi_result(rd.read_data_from_resource("dataset2")[0],
                              rd.read_data_from_resource("dataset2")[1],
                              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                              [[[-1 / 7, -2, 4.5],
                                [-3, 1, 0],
                                [0, 0, 0]],
                               [[-1, -1, 1.8],
                                [-0, 1.1, 0],
                                [0, 1, 0]],
                               [[-1, -1, 1.7],
                                [-0, -1.2, 0],
                                [-2, 1, 0]]
                               ], 4)

    # show_softmax_binary_result(rd.read_data_from_resource("dataset1")[0],
    #                   rd.read_data_from_resource("dataset1")[1],
    #                   [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    #                   [[[-1 / 7, -2],
    #                     [-3, 1],
    #                     [-3, 1]],
    #                    [[-1, -1],
    #                     [-0, 1.1],
    #                     [-3, 1]],
    #                    [[-1, -1],
    #                     [-0, -1.2],
    #                     [-2, 1]]
    #                    ], 4)


if __name__ == "__main__":
    test_show_multi_result()
