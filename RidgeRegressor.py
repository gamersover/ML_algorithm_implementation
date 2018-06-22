# coding:utf-8
"""
岭回归算法
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RidgeRegression(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.Y = y.reshape(-1,1)
        self.calW()

    def calW(self):
        n = self.X.shape[0]
        feat = np.hstack([np.ones([n,1]), self.X])
        m = feat.shape[1]
        self.w = np.linalg.inv(np.dot(feat.T, feat) + self.k*np.eye(m)) @ feat.T
        self.w = self.w @ self.Y

    def predict(self, x):
        return x @ self.w


if __name__ == "__main__":
    x, y = [], []
    for i in np.linspace(0,1,10):
        for j in np.linspace(0,1,10):
            x.append([i, j])
            y.append(2.3 * i + 4.2 * j + 1.3 + np.random.random())
    x = np.array(x)
    y = np.array(y)
    lr = RidgeRegression(2)
    lr.fit(x, y)
    print(lr.w)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x[:,0], x[:,1], y)
    # plt.show()
