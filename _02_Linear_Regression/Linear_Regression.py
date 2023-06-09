# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X, y = read_data()
    alpha = -0.1
    XT_X = np.dot(X.T, X)
    I = np.identity(XT_X.shape[0])
    weight = np.dot(np.dot(np.linalg.inv(XT_X + alpha * I), X.T), y)
    return weight @ data



def lasso(data):
    X,y = read_data()
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y))
    return weight @ data


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
