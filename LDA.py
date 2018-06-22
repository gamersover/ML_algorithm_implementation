#coding:utf-8
"""
LDA:线性判别分析算法实现
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class LDA(object):

	def __init__(self, n_components):
		self.n_components = n_components

	def fit(self, x, y):
		self.data = x
		self.label = y
		assert len(label.shape) == 2
		assert label.shape[1] == 1
		return self._lda()

	def _lda(self):
		sample_mu = np.mean(self.data, axis=0)
		classes = np.unique(self.label)

		n_classes = classes.shape[0]
		n_features = data.shape[1]
		if self.n_components > n_features-1:
			self.n_components = n_features-1
			print("warning:parameter n_components larger than feature_size, defalut set as feture_size-1")

		sw = np.zeros([n_features, n_features])
		sb = np.zeros([n_features, n_features])
		for i in range(n_classes):
			split_data = data[label[:,0]==classes[i], :]
			N = split_data.shape[0]
			class_mu = np.mean(split_data, axis=0)
			w = split_data - class_mu
			v = (class_mu - sample_mu).reshape(1,-1)
			sw += np.dot(w.T, w)
			sb += np.dot(v.T, v) * N

		J = np.dot(np.linalg.inv(sw), sb)
		self.eigvalue, self.eigvector = np.linalg.eig(J)
		self.reserved = self.eigvector[:,np.argsort(self.eigvalue)][:,-self.n_components:]
		return np.dot(self.data, self.reserved)

	def get_direction(self):
		return self.reserved[:,-1].T


if __name__ == '__main__':
	np.random.seed(100)
	x = np.random.randn(100, 3)
	data1 = 0.2 * x + np.array([0,0,0])
	data2 = 0.2 * x + np.array([1,1,1])

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(data1[:,0], data1[:,1], data1[:,2])
	ax.scatter(data2[:,0], data2[:,1], data2[:,2])
	data = np.vstack((data1, data2))
	label = np.vstack((np.zeros([100,1]), np.ones([100,1])))
	lda = LDA(2)
	lda_y = lda.fit(data, label)
	d = lda.get_direction()
	print(d)
	# ax.plot([0, d[0,0]], [0, d[0,1]], [0, d[0,2]])
	ax.plot([0, d[0]], [0, d[1]], [0, d[2]])
	plt.show()