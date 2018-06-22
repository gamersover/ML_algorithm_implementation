#coding=UTF-8
'''
统计学习方法
第四章    朴素贝叶斯

'''
import numpy as np
from collections import Counter

class NB:
    '''
    function fit,predict  you can use
    function get_cla_index,get_con_pro is inner func that you can not use
    '''
    
    def fit(self, x, y, lamda=1):
        '''
        x:  feature_x
        y；    label_y
        lamda:  贝叶斯估计中的平滑参数，当lamda=0时为朴素贝叶斯; 当lamda=1时为laplace平滑
        classes:  label_y的去重后的列表
        feature_value:  feature_x中每个特征可能取的值
        cla_index:   classes中每个类别包含的feature_x的index
        can_pro:    贝叶斯估计中的条件概率
        prior_pro:   贝叶斯估计中的先验概率
        '''
        self.x = x
        self.y = y
        self.lamda = lamda
        self.classes = list(set(y))
        self.feature_value = [set(x[:,i]) for i in range(x.shape[1])]
        self.cla_index = self.get_cla_index()
        self.con_pro = self.get_con_pro()   
        self.prior_pro = [(Counter(y)[i]+self.lamda)/(len(y)+self.lamda*len(self.classes)) for i in self.classes]
    
    def get_cla_index(self):
        cla_index = []
        for i in self.classes:
            x_class = []
            for j in range(self.x.shape[0]):
                if self.y[j] == i:
                    x_class.append(j)
            cla_index.append(x_class)
        return cla_index
    
    def get_con_pro(self):
        con_pro = []
        for i in range(len(self.classes)):
            value = []
            for j in range(self.x.shape[1]):   
                li = [self.x[index][j] for index in self.cla_index[i]]
                feature_dic = {}
                for k in self.feature_value[j]:
                    feature_dic[k] = (Counter(li).get(k,0)+self.lamda)/(len(li)+self.lamda*len(self.feature_value[j]))
                value.append(feature_dic)
            con_pro.append(value)
        return con_pro
        
    
    def predict(self, x_pre):
        max_pro = 0
        for i in range(len(self.classes)):
            pre_pro = self.prior_pro[i]
            for j in range(self.x.shape[1]):
                pre_pro *= self.con_pro[i][j][str(x_pre[j])]
            if pre_pro >= max_pro:
                max_pro = pre_pro
                y_pre = self.classes[i]
        return y_pre

if __name__ == '__main__':
    x = np.array([[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],[2,'M'],[2,'M'],
              [2,'L'],[2,'L'],[3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    x_pre = [2,'S']
    lamda = 1
    clf = NB()
    clf.fit(x,y,lamda)
    y_pre = clf.predict(x_pre)
    print(y_pre)

        
        

