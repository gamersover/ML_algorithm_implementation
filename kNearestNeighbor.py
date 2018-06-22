#coding=UTF-8
'''
统计学习方法
第三章  k近邻算法
'''
import numpy as np
from collections import Counter

class KNN:
    def fit(self,x,y):
        self.x = x
        self.y = y
        
    def get_p_distance(self,x_1,x_2,p): 
        return np.sum((x_1-x_2)**p)**(1/p)
    
    def predict(self,x_pre,k=3,p=2):
        distance = [(i,self.get_p_distance(self.x[i], x_pre,p)) for i in range(self.x.shape[0])]
        distance.sort(key=lambda t:t[1])
        classes = [self.y[i[0]] for i in distance[:k]]
        return Counter(classes).most_common(1)[0][0]

if __name__ =='__main__':
    x = np.array([[3,3],[4,3],[1,1],[3,5]])
    y = np.array([1,1,0,0])
    x_pre = [2,2]
    clf = KNN()
    clf.fit(x,y)
    y_pre = clf.predict(x_pre,k=3,p=1)
    print(y_pre)
    

            

