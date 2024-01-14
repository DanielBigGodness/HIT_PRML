import pandas as pd
import numpy as np
# 1 导入数据集
dataSet = pd.read_csv('logistic_data1.txt',header = None)
dataSet.columns =['x0','x1','y']
print(dataSet)

# 2 定义Sigmoid函数
def sigmoid(inX):
    s = 1/(1+np.exp(-inX))
    return s

# 3 定义归一化函数
"""
函数功能：归一化（期望为0，方差为1）
参数说明：
    xMat：特征矩阵
返回：
    inMat：归一化之后的特征矩阵
"""
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat,axis = 0)
    inVar = np.std(inMat,axis = 0)
    inMat = (inMat - inMeans)/inVar
    return inMat

# 4 使用批量梯度下降法
"""
函数功能：使用BGD求解逻辑回归
参数说明：
dataSet：DF数据集
alpha：步长
maxCycles：最大迭代次数
返回：
weights：各特征权重值
"""
def BGD_LR(dataSet,alpha,maxCycles):
    xMat = np.mat(dataSet.iloc[:,:-1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    xMat = regularize(xMat)
    m,n = xMat.shape
    weights = np.zeros((n,1))
    for i in range(maxCycles):
        grad = xMat.T * (sigmoid(xMat * weights) - yMat) / m
        weights = weights -alpha*grad
    return weights


# 5 准确率计算
ws=BGD_LR(dataSet,alpha=0.005,maxCycles=5000)
xMat = np.mat(dataSet.iloc[:, :-1].values)
yMat = np.mat(dataSet.iloc[:, -1].values).T
inMat = xMat.copy()
xMat = regularize(xMat)

p = sigmoid(xMat * ws).A.flatten()
# ans: p<0.5
# axx0 + bxx1  <  0
# bxx1 < - axx0
# xx1 < -a/b xx0
# xx = (x -mean) / std
# (x1-mean_x1)/sd_x1 < -a/b (x0-mean_x0)/std_x0
# x1 - mean_x1 <  -(a*std_x1)/(b*std_x0)(x0-mean_x0)
# x1 < -(a*std_x1)/(b*std_x0)x0 + (a*std_x1)/(b*std_x0)mean_x0 + mean_x1
inMeans = np.mean(inMat, axis=0)
inVar = np.std(inMat, axis=0)
std_x0 = inVar[:, :1]
std_x1 = inVar[:, 1:]
mean_x0 = inMeans[:, :1]
mean_x1 = inMeans[:, 1:]
a = ws[0]
b = ws[1]
wn = -(a*std_x1)/(b*std_x0)
bn = (a*std_x1)/(b*std_x0)*mean_x0 + mean_x1
wn = np.array(wn)
w = wn[0][0]
bn = np.array(bn)
b = bn[0][0]

for i, j in enumerate(p):
    if j < 0.5:
        p[i] = 0
    else:
        p[i] = 1

train_error = (np.fabs(yMat.A.flatten() - p)).sum()
train_error_rate = train_error / yMat.shape[0]
trainAcc = 1-train_error_rate
print("trainAcc=",trainAcc)

# 6  可视化
import matplotlib.pyplot as plt
import os
import pandas as pd
data_file = os.path.join('.', 'logistic_data1.csv')
data = pd.read_csv(data_file, )
data_0 = data[data['y']==0]
data_1 = data[data['y']==1]

x0 = np.linspace(30, 100)
x1 = w*x0 +b
plt.plot(x0, x1)

plt.scatter(data_0['x0'], data_0['x1'], c='#1f77b4')  # blue
plt.scatter(data_1['x0'], data_1['x1'], c='#ff7f0e')  # orange
plt.legend(['y=wx+b', 'y = 0','y = 1'])

plt.show()
