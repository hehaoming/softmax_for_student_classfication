# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import read_data
import show_results

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
    w = np.zeros([x.shape[1], len(np.unique(y))])
    learningRate = 1e-5
    losses = []
    thetaList = []
    for i in range(iterations):
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        losses_per_epoch = []
        for id in idx :
            xx = np.reshape(x[id], (1, x.shape[1]))
            yy = np.reshape(y[id], (1))
            loss, grad = getLoss(w, xx, yy)
            losses_per_epoch.append(loss)
            w = w - (learningRate * grad)
        losses.append(np.mean(losses_per_epoch))
        thetaList.append(w)
    return w, losses, thetaList

def getProbsAndPreds(someX, w):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def getAccuracy(someX, someY, w):
    prob,prede = getProbsAndPreds(someX, w)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

if __name__ == "__main__":
    x, y = read_data.read_data_from_resource()
    w, losses, thetaList = softmaxRegressionSGD(x, y, 100000)
    print('Training Accuracy: ', getAccuracy(x, y, w))
    plt.plot(losses)
    plt.legend()  # 将样例显示出来
    plt.show()
    #show_results.show_result(train_data=x, train_labels=y,error_list= losses,theta_list= thetaList, iterator=1000)
