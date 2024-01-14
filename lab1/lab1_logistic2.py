import pandas as pd
import numpy as np
# 1 导入数据集
dataSet = pd.read_csv('logistic_data2.txt',header = None)
dataSet.columns =['x0','x1','y']

xMat = np.array(dataSet.iloc[:, :-1].values)
Mat = np.array(xMat)

new_col = []
for i in range(0, 118):
    new_col.append(float(Mat[i][0]) * float(Mat[i][1]))
xMat = np.insert(xMat, 2, new_col, axis=1)  # 在第三列插入新列
new_col = []
for i in range(0, 118):
    new_col.append(float(Mat[i][0]) * float(Mat[i][0]))
xMat = np.insert(xMat, 3, new_col, axis=1)  # 在第四列插入新列
new_col = []
for i in range(0, 118):
    new_col.append(float(Mat[i][1]) * float(Mat[i][1]))
xMat = np.insert(xMat, 4, new_col, axis=1)  # 在第五列插入新列


# 2 定义Sigmoid函数
def sigmoid(inX):
    s = 1/(1+np.exp(-inX))
    return s

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
def BGD_LR(dataSet, alpha, maxCycles):
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    m,n = xMat.shape
    weights = np.zeros((n, 1))
    bias = np.zeros((1,1))
    nom = np.ones(118).T
    for i in range(maxCycles):
        gradb = np.dot(nom,(sigmoid(np.dot(xMat, weights)+ bias ) - yMat))
        print(gradb)
        grad = np.dot(xMat.T,(sigmoid(np.dot(xMat, weights)+ bias ) - yMat))          #损失函数求梯度
        weights = weights -alpha * grad
        bias = bias -alpha * gradb
        #print("i=",i,weights)
    return weights,bias

def BGD_LR_L2(dataSet, alpha, lamba, maxCycles):
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    m,n = xMat.shape
    weights = np.zeros((n, 1))
    bias = np.zeros((1,1))
    nom = np.ones(118).T
    gradb = np.ones((1,1))
    grad = np.ones((1, 1))
    for i in range(maxCycles):
        print(gradb)
        gradb = (np.dot(nom,(sigmoid(np.dot(xMat, weights)+ bias ) - yMat)) + lamba * gradb)
        grad = (np.dot(xMat.T,(sigmoid(np.dot(xMat, weights)+ bias ) - yMat)) + lamba * grad )
        #损失函数求梯度
        weights = weights -alpha * grad
        bias = bias -alpha * gradb
        #print("i=",i,weights)
    return weights,bias

# 5 准确率计算
ws,bs=BGD_LR(dataSet,alpha=0.05,maxCycles=4800)
# print(bs)
yMat = np.mat(dataSet.iloc[:, -1].values).T
p = sigmoid(np.dot(xMat, ws)+ bs).A.flatten()
ws = np.array(ws)
bs = np.array(bs)
a1 = ws[0][0]
a2 = ws[1][0]
a3 = ws[2][0]
a4 = ws[3][0]
a5 = ws[4][0]
b = bs[0][0]
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
data_file = os.path.join('.', 'logistic_data2.csv')
data = pd.read_csv(data_file, )
data_0 = data[data['y']==0]
data_1 = data[data['y']==1]

x0 = np.linspace(-0.75, 1)
x1 = np.linspace(-0.75, 1)

# 转化为网格
x0,x1=np.meshgrid(x0,x1)
z= a1*x0 + a2*x1 + a3*x0*x1 +a4*x0*x0 +a5*x1*x1 + b
plt.contour(x0,x1,z,0)

plt.scatter(data_0['x0'], data_0['x1'], c='#1f77b4')  # blue
plt.scatter(data_1['x0'], data_1['x1'], c='#ff7f0e')  # orange
plt.legend(['y = 0', 'y = 1'])

plt.show()

ws,bs=BGD_LR_L2(dataSet,alpha=0.05,lamba=-0.65, maxCycles=4800)
# print(bs)
yMat = np.mat(dataSet.iloc[:, -1].values).T
p = sigmoid(np.dot(xMat, ws)+ bs).A.flatten()
ws = np.array(ws)
bs = np.array(bs)
a1 = ws[0][0]
a2 = ws[1][0]
a3 = ws[2][0]
a4 = ws[3][0]
a5 = ws[4][0]
b = bs[0][0]
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
data_file = os.path.join('.', 'logistic_data2.csv')
data = pd.read_csv(data_file, )
data_0 = data[data['y']==0]
data_1 = data[data['y']==1]

x0 = np.linspace(-0.75, 1)
x1 = np.linspace(-0.75, 1)

# 转化为网格
x0,x1=np.meshgrid(x0,x1)
z= a1*x0 + a2*x1 + a3*x0*x1 +a4*x0*x0 +a5*x1*x1 + b
plt.contour(x0,x1,z,0)

plt.scatter(data_0['x0'], data_0['x1'], c='#1f77b4')  # blue
plt.scatter(data_1['x0'], data_1['x1'], c='#ff7f0e')  # orange
plt.legend(['y = 0', 'y = 1'])

plt.show()
