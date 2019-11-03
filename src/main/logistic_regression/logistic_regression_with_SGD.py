import numpy as np
from commons.read_data import read_data_from_resource
from show_results import show_binary_result
from commons.utils import one_hot_encoding

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
        loss += -classLabels[i] * np.log(propability) - (1 - classLabels[i]) * np.log(1 - propability)
        print("dataMat[i]", dataMat[i])
        print("weights", (weights))
        print("sum_theta_x", sum_theta_x)
        print("loss", loss)
    return loss


def h(x, w):
    return np.dot(x, w.T)

def cost_and_grad(x, y, w, if_grad=True):
    grad = np.zeros(w.shape) * np.nan
    y_one_hot = np.zeros(2)
    y_one_hot[y] = 1
    front_f_x = np.dot(x, w)
    y_hat = sigmoid(front_f_x)
    h_x = np.concatenate((y_hat, 1 - y_hat), axis=-1)
    # 平均cost
    cost = - 1 / len(y) * sum(sum(y_one_hot * np.log(h_x)))
    if if_grad:
        # 平均梯度
        grad = - 1 / len(y) * np.dot(x.transpose(), y_one_hot - y_hat)
    return cost, grad
# 随机梯度下降
# 随机梯度下降每次更新权重只需要一个样本
def stochastic_grad_descent(x, y, w=None, lr=0.001, epochs=1000):
    alpha = 0.001
    if w == None:
        w = np.ones((x.shape[1], 1))
    loss_list = []
    theta_list = []
    for iter in range(epochs):
        index = np.random.randint(1, x.shape[0])
        w = w - lr * cost_and_grad(np.array(x[index]), np.array(y[index]), w)[1]
        theta_list.append(w)
        loss_list.append(cost_and_grad(x, y, w, False)[0])
        # error = classLabels[i] - h
        # new_weights = weights + alpha * error * dataMatIn[i]
        # loss = loss_funtion(dataMatIn, classLabels, weights)
        # new_loss = loss_funtion(dataMatIn, classLabels, new_weights)
        # loss_array.append(new_loss)
        # theta_array.append(list(np.array(new_weights).squeeze()))
        # print("iter_count: ", iter_count, "the loss:", loss)
        # iter_count += 1

    # return weights, loss_array, theta_array


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
stochastic_grad_descent(data, labels)

# print("theta", theta_array)
# print("loss", loss_array)

# show_binary_result(data, labels, loss_array, theta_array, 1000)
