#coding=UTF-8
'''
统计学习方法
第二章 感知机
train_1函数为感知机模型学习算法
train_2函数为感知机对偶形式学习算法
'''
import numpy as np

def train_1(x,y,eta=1):
    w = np.zeros(x.shape[1])
    b = 0
    while True:
        flag = 0
        for i in range(x.shape[0]):
            if y[i]*(np.dot(x[i],w) + b) <= 0:
                if flag==0:
                    flag = 1
                w += eta*x[i]*y[i]
                b += eta*y[i]
                break
        if flag == 0:
            break
    
    return w,b    

def train_2(x,y,eta=1):
    alpha = np.zeros(x.shape[0])
    b = 0
    Gram = np.dot(x, x.T)
    while True:
        flag = 0
        for i in range(x.shape[0]):
            if y[i]*(np.dot(alpha*y,Gram[:,i])+b) <= 0:
                alpha[i] += eta
                b += eta * y[i]
                if flag == 0:
                    flag = 1
                break
        if flag == 0:
            break
    w = np.dot(alpha*y,x)
    return w,b    
    
    
    
if __name__ == '__main__':
    x = np.array([[3,3],[4,3],[1,1]])
    y = np.array([1,1,-1])
    w,b = train_2(x,y)
    print(w,b)