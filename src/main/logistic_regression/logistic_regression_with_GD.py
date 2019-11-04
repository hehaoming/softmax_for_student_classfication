import numpy as np
from read_data import read_data_from_resource
import matplotlib.pyplot as plt
from show_results import show_binary_result


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def loss_funtion(dataMat, classLabels, weights):
    m, n = np.shape(dataMat)
    loss = 0.0
    for i in range(m):
        sum_theta_x = 0.0
        for j in range(n):
            sum_theta_x += dataMat[i, j] * weights.T[0, j]
        propability = sigmoid(sum_theta_x)
        loss += -classLabels[i] * np.log(propability) - (1 - classLabels[i]) * np.log(1 - propability)
    return loss * 1.0 / 80.0


def gradAscent(dataMatIn, classLabels):
    loss_array = []
    theta_array = []
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 5000
    # weights=np.ones((n,1))  #0.41
    # weights = np.random.normal(loc=0, scale=0.01, size=(n, 1)) #0.40
    weights = np.zeros((n, 1))  # 0.4073
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        loss = loss_funtion(dataMatrix, labelMat, weights)
        loss_array.append(np.array(loss).squeeze())
        theta_array.append(np.array(weights).squeeze())
    return np.array(weights), loss, loss_array, theta_array


def plotBestFit(dataArr, labelMat, weights):
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-0.1, 1.0, 0.01)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X');
    plt.ylabel('Y')
    plt.show()


dataMat, labelMat = read_data_from_resource()
# minmax正则化：否则数据不好收敛，到nan
min1 = np.min(dataMat[:, 1])
max1 = np.max(dataMat[:, 1])
min2 = np.min(dataMat[:, 2])
max2 = np.max(dataMat[:, 2])
m, n = np.shape(dataMat)
for i in range(m):
    dataMat[i][1] = (dataMat[i][1] - min1) / (max1 - min1)
    dataMat[i][2] = (dataMat[i][2] - min2) / (max2 - min2)
dataArr = np.array(dataMat)
weights, losst, loss_array, theta_array = gradAscent(dataArr, labelMat)
print("weight:", weights)
print("loss:", losst)
# 数据可视化
plotBestFit(dataArr, labelMat, weights)

# print("loss:", loss_array)
# print("theta:", theta_array)

plt.plot(loss_array)
plt.show()
show_binary_result(dataMat,
                   labelMat,
                   loss_array,
                   theta_array, 5000)
