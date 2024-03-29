"""多分类SGD
@author: hehaoming
@time: 2019/11/2 11:03
"""

import numpy as np
import logging
from commons.read_data import read_data_from_resource
import utils
from commons.show_results import show_softmax_multi_result

class SoftmaxClassifier:
    def __init__(self, num_classes):
        self.w = None
        self.w_list = []
        self.cost_list = []
        self.num_classes = num_classes

    def cost_and_grad(self, x, y, w, if_grad=True):
        grad = np.zeros(w.shape) * np.nan
        y_one_hot = utils.one_hot_encoding(y, self.num_classes)
        front_f_x = np.dot(x, w)
        y_hat = utils.softmax(front_f_x)
        # 平均cost
        cost = - 1 / len(y) * sum(sum(y_one_hot * np.log(y_hat)))
        if if_grad:
            # 平均梯度
            grad = - 1 / len(y) * np.dot(x.transpose(), y_one_hot - y_hat)
        return cost, grad

    def fit(self, x, y, w=None, lr=0.01, epochs=1000):
        if w is None:
            w = np.zeros((x.shape[1], self.num_classes))

        # 记录初始状态
        self.w_list.append(w)
        self.cost_list.append(self.cost_and_grad(x, y, w, False)[0])

        # 迭代更新参数
        for i in range(epochs):
            logging.debug("epoch: %d", i)
            # 随机化
            p = np.random.randint(0, len(y))
            w = w - lr * self.cost_and_grad(np.array([x[p]]), np.array([y[p]]), w)[1]
            self.w_list.append(w)
            self.cost_list.append(self.cost_and_grad(x, y, w, False)[0])
        self.w = w

    def score(self, x, y):
        prediction = self.predict(x)
        acc = 1 / len(y) * sum([(prediction[i] == y[i]) for i in range(len(y))])
        return acc

    def predict(self, x):
        out = np.array([np.argmax(x) for x in utils.softmax(x.dot(self.w))])
        return out


if __name__ == "__main__":
    eopch = 5000
    num_class = 3
    softmaxClassifier = SoftmaxClassifier(3)
    data = read_data_from_resource("dataset2")
    # min1 = np.min(data[0][:, 1])
    # max1 = np.max(data[0][:, 1])
    # min2 = np.min(data[0][:, 2])
    # max2 = np.max(data[0][:, 2])
    # m, n = np.shape(data[0])
    # for i in range(m):
    #     data[0][i][1] = (data[0][i][1] - min1) / (max1 - min1)
    #     data[0][i][2] = (data[0][i][2] - min2) / (max2 - min2)
    w = np.zeros((data[0].shape[1], num_class))
    # w = np.ones((data[0].shape[1], num_class))
    # w = np.random.normal(0, 1, (data[0].shape[1], num_class))
    softmaxClassifier.fit(data[0], data[1], w=w, lr=0.05, epochs=eopch)
    print("weight:\n", softmaxClassifier.w)
    print("cost:", softmaxClassifier.cost_list[-1])
    print("acc:", softmaxClassifier.score(data[0], data[1]))
    show_softmax_multi_result(data[0], data[1], softmaxClassifier.cost_list, softmaxClassifier.w_list, eopch)