"""工具函数
@author: hehaoming
@time: 2019/11/2 11:03
"""
import numpy as np


def softmax(x):
    e = np.exp(x)
    sum_e_x = sum(e.T)
    result = np.array([e[i] / sum_e_x[i] for i in range(len(x))])
    return result


def one_hot_encoding(y, num_class):
    n = y.shape[0]
    y_one_hot = np.zeros((n, num_class))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot
