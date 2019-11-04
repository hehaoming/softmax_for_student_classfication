"""读取数据集数据
@author: litian
@time: 2019/11/4 21:00
"""
import numpy as np
from read_data import read_data_from_resource
from show_results import show_binary_result


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 损失函数
def loss_funtion(dataMat, classLabels, weights):
    m, n = np.shape(dataMat)
    loss = 0.0
    for i in range(m):
        sum_theta_x = 0.0

        sum_theta_x += np.dot(dataMat[i], (weights))
        propability = sigmoid(sum_theta_x)
        loss += -(1 / m) * (classLabels[i] * np.log(propability) + (1 - classLabels[i]) * np.log(1 - propability))
        # print("dataMat[i]", dataMat[i])
        # print("weights", (weights))
        # print("sum_theta_x", sum_theta_x)
        # print("loss", loss)
    return loss


# 随机梯度下降
# 随机梯度下降每次更新权重只需要一个样本
def stocGradDescent(dataMatIn, classLabels, max_iters):
    alpha = 0.03
    m, n = np.shape(dataMatIn)  # m是第一维度的元素个数，即样本数，n是第二维度元素个数，即特征数
    # weights = np.ones(n)
    weights = [0, 0, 0]
    loss_array = []
    theta_array = []
    # max_iters = 10000  # 最大迭代次数
    iter_count = 0  # 当前迭代次数

    while iter_count < max_iters:
        loss = 0
        random = np.random.randint(1, m)
        h = sigmoid(np.sum(weights * dataMatIn[random]))
        error = classLabels[random] - h
        new_weights = weights + alpha * error * dataMatIn[random]
        weights = new_weights
        # for i in range(m):
        #     h = sigmoid(np.sum(new_weights * dataMatIn[i]))
        #     loss = loss_funtion(dataMatIn, classLabels, weights)
        h = sigmoid(np.sum(new_weights * dataMatIn[random]))
        loss = loss_funtion(dataMatIn, classLabels, weights)
        loss_array.append(loss)
        theta_array.append(weights)
        iter_count += 1

    return weights, loss_array, theta_array


data, labels = read_data_from_resource("dataset1")
# minmax正则化：否则数据不好收敛，到nan
min1 = np.min(data[:, 1])
max1 = np.max(data[:, 1])
min2 = np.min(data[:, 2])
max2 = np.max(data[:, 2])
m, n = np.shape(data)
for i in range(m):
    data[i][1] = (data[i][1] - min1) / (max1 - min1)
    data[i][2] = (data[i][2] - min2) / (max2 - min2)
max_iters = 10000
weight, loss_array, theta_array = stocGradDescent(data, labels, max_iters)
print("theta is :", theta_array[-1])

print("loss", loss_array[-1])

show_binary_result(data, labels, loss_array, theta_array, max_iters)
