from read_data import read_data_from_resource
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from show_results import show_multi_result

def softmax(z):
    log_c = np.max(z, axis = 1) * (-1)
    log_c = log_c.reshape(-1, 1)
    prob = np.exp(z + log_c)
    prob = prob / np.exp(z + log_c).sum(axis = 1).reshape(-1, 1)
    return np.clip(prob, 1e-15, 1-1e-15)

#损失定义为交叉熵的均值
def compute_cost(X, y, weight):
    z=np.dot(X,weight)
    activation = softmax(z)
    cross_entropy = - np.sum(np.log(activation) * (y), axis = 1)
    return np.mean(cross_entropy)

def one_hot_encoder(labels,m):
    coder = np.zeros((m, 2))
    for i in range(m):
        coder[i, labels[i]] = 1
    return coder

def softmax_gradient_descent(X, y, epochs=10000, learning_rate=0.01):
    weight = np.random.normal(loc=0, scale=0.01, size=(X.shape[1], 2))
    cost_array = []
    weight_array=[]
    for epoch in range(epochs):
        z = np.dot(X, weight)
        activation = softmax(z)
        diff = activation - y
        grad = np.dot(X.T, diff)
        weight -= learning_rate * grad
        cost = compute_cost(X, y, weight)
        cost_array.append(cost)
        weight_array.append(np.array(weight).squeeze())
    return np.array(weight[:,1]).squeeze(), cost_array,weight_array


#这里修改了: data.insert(0,1.0)
data,label=read_data_from_resource()
label=np.array(label)



#minmax正则化：否则数据不好收敛，到nan
min1=np.min(data[:,1])
max1=np.max(data[:,1])
min2=np.min(data[:,2])
max2=np.max(data[:,2])
m,n=np.shape(data)

for i in range(m):
    data[i][1]=(data[i][1]-min1)/(max1-min1)
    data[i][2]=(data[i][2]-min2)/(max2-min2)

labels=one_hot_encoder(label,m)

learned_w, cost_array,weight_array= softmax_gradient_descent(data,labels)
print("Learned weights:")
print(learned_w)

#用作测试数据正确
plt.plot(range(len(cost_array)), cost_array)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()

#传出参数
print("loss：",cost_array)
print("weight：",weight_array)

#参数尺度
print("loss大小：",np.shape(cost_array))
print("weight大小：",np.shape(weight_array))

#尝试绘出图像
show_multi_result(data,
                label,
                cost_array,
                weight_array, 10000,2)