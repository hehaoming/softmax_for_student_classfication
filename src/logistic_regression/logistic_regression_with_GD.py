import numpy as np
import matplotlib.pyplot as plt
from read_data import read_data_from_resource #修改部分train_data.append([float(1), float(item[x1_index]), float(item[x2_index])])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_funtion(dataMat, classLabels, weights):
    m, n = np.shape(dataMat)
    loss = 0.0
    for i in range(m):
        sum_theta_x = 0.0
        for j in range(n):
            sum_theta_x += dataMat[i, j] * weights.T[0, j]
        propability = sigmoid(sum_theta_x)
        loss += -classLabels[i] * np.log(propability) - (1 - classLabels[i]) * np.log(1 - propability)
    return loss


def logistic_grad_descent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  #(m,n)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    alpha = 0.001
    maxstep = 10000
    eps = 0.001
    count = 0
    loss_array = []
    theta_array=[]

    for i in range(maxstep):
        loss = loss_funtion(dataMatrix, labelMat, weights)
        h_theta_x = sigmoid(dataMatrix * weights)
        e = h_theta_x - labelMat
        new_weights = weights - alpha * dataMatrix.T * e
        new_loss = loss_funtion(dataMatrix, labelMat, new_weights)
        loss_array.append(new_loss)
        # if abs(new_loss - loss) < eps:
        #     break
        # else:
        #     weights = new_weights
        #     count += 1
        if i%1000==0:#进度
            print(i/1000)
        weights = new_weights
        count += 1
        theta_array.append(list(np.array(new_weights).squeeze()))

    print("count is: ", count)
    print("loss is: ", loss)
    print("weights is: ", weights)

    return weights, loss_array , theta_array

def plotloss(loss_array):
    n = len(loss_array)
    plt.xlabel("times")
    plt.ylabel("loss")
    plt.scatter(range(1, n+1), loss_array)
    plt.show()

def plottheta(data,labels,r):
    plt.scatter(data[labels == 0][:, 1], data[labels == 0][:, 2], color='green', label='class A')
    plt.scatter(data[labels == 1][:, 1], data[labels == 1][:, 2], color='red', label='class B')
    plt.legend()
#这里修改了: data.insert(0,1.0)
data,labels=read_data_from_resource()
#minmax正则化：否则数据不好收敛，到nan
min1=np.min(data[:,1])
max1=np.max(data[:,1])
min2=np.min(data[:,2])
max2=np.max(data[:,2])
m,n=np.shape(data)
for i in range(m):
    data[i][1]=(data[i][1]-min1)/(max1-min1)
    data[i][2]=(data[i][2]-min2)/(max2-min2)
r, loss_array,theta_array = logistic_grad_descent(data, labels)
#r = np.mat(r).transpose()
#画出最终结果图
plotloss(loss_array)
plottheta(data,labels,r)
#传出参数
print("theta",theta_array)
print("loss",loss_array)
#迭代次数
print("num",np.shape(loss_array)[0])
#之后每次绘图时使用参数
#以第一次为例
print(theta_array[0],loss_array[0])



