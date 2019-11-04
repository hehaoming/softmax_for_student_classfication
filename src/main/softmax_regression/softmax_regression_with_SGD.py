# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import read_data
import show_results
import utils

def oneHotIt(y):
    K = 2
    eyes_mat = np.eye(K)  # 按分类数生成对角线为1的单位阵
    y_onehot = np.zeros((y.shape[0], K))  # 初始化y的onehot编码矩阵
    for i in range(0, y.shape[0]):
        y_onehot[i] = eyes_mat[y[i]]  # 根据每行y值，更新onehot编码矩阵
    return y_onehot

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getLoss(w,x,y):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) #And compute the gradient for that loss
    return loss,grad

def softmaxRegressionSGD(x, y, iterations):
    # w = np.zeros([x.shape[1], len(np.unique(y))])
    w = np.random.rand(x.shape[1], len(np.unique(y)))
    learningRate = 0.03
    losses = []
    thetaList = []
    for i in range(iterations):
        id = np.random.randint(0, x.shape[0])
        xx = np.reshape(x[id], (1, x.shape[1]))
        yy = np.reshape(y[id], (1))
        loss, grad = getLoss(w, xx, yy)
        w = w - (learningRate * grad)
        loss2, grad2 = getLoss(w, x, y)
        thetaList.append(w)
        losses.append(loss2)
    return w, losses, thetaList

def getProbsAndPreds(someX, w):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def getAccuracy(someX, someY, w):
    prob,prede = getProbsAndPreds(someX, w)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

def predict(x, w):
    out = np.array([np.argmax(x) for x in softmax(x.dot(w))])
    return out
def score(x, y):
    prediction = predict(x)
    acc = 1 / len(y) * sum([(prediction[i] == y[i]) for i in range(len(y))])
    return acc
if __name__ == "__main__":
    data, label = read_data.read_data_from_resource()
    # minmax正则化：否则数据不好收敛，到nan
    min1 = np.min(data[:, 1])
    max1 = np.max(data[:, 1])
    min2 = np.min(data[:, 2])
    max2 = np.max(data[:, 2])
    m, n = np.shape(data)
    for i in range(m):
        data[i][1] = (data[i][1] - min1) / (max1 - min1)
        data[i][2] = (data[i][2] - min2) / (max2 - min2)

    w, losses, thetaList = softmaxRegressionSGD(data, label, 5000)
    print('Training Accuracy: ', getAccuracy(data, label, w))
    # plt.plot(losses)
    # plt.legend()  # 将样例显示出来
    # plt.show()
    # los2 = losses[:1000]
    # the2 = thetaList[:1000]
    # show_results.show_softmax_binary_result(train_data=x, train_labels=y,error_list= los2,theta_list= the2, iterator=1000)
    # los2 = losses[-1000:]
    # the2 = thetaList[-1000:]
    # show_results.show_softmax_binary_result(train_data=data, train_labels=label,error_list= los2,theta_list= the2, iterator=1000)
    show_results.show_softmax_binary_result(train_data=data, train_labels=label, error_list=losses, theta_list=thetaList, iterator=5000)
