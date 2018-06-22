#coding=UTF-8
'''
k_means算法的实现
'''
import numpy as np
from matplotlib import pyplot as plt
import random

class K_means:
    
    def fit(self, x, k, iter_num):
  
        init_centr = random.sample(list(x),k)
    
        for i in range(iter_num):
            new_cla = self.get_cla(x, init_centr)
            cla_re = self.get_li(k, new_cla)
            new_centr = [self.update_centr(i) for i in cla_re]
            if np.array_equal(init_centr,new_centr):
                break
            init_centr = new_centr

            
            
        return init_centr, cla_re
        
    def get_dis(self,x,y):
        return sum((x-y)**2)
    
    def update_centr(self,idx_li):
        sum = np.zeros(x.shape[1])
        for i in idx_li:
            sum += x[i]/len(idx_li)
        return sum
    
    def get_cla(self,x,init_centr):
        new_cla = []
        for i in x:
            d = [self.get_dis(i, j) for j in init_centr]
            new_cla.append(np.argmin(d))
        return new_cla
    
    def get_li(self,k,new_cla):
        a_li = []
        for i in range(k):
            li = []
            for j in range(len(new_cla)):
                if new_cla[j] == i:
                    li.append(j)
            a_li.append(li)
        return a_li

x = np.array([[1,5],[1,4],[2,4],[2,4.5],[3,-1],[4,1],[4,0.8],[4,0.8],
             [4.2,3],[5,1]])

plt.scatter(x[:,0],x[:,1])
# plt.show()



centr, cla_re = K_means().fit(x, k=2, iter_num=5)
print(centr)
print(cla_re)


            
        
