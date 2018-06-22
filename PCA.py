#coding:utf-8
"""
pca算法实现
"""
import numpy as np

class PCA(object):

	def __init__(self, n_components, verbose=0):
		self.n_components = n_components
		self.verbose = verbose
		assert self.verbose in (0,1)

	def fit(self, x):
		self.x = x
		if self.n_components > x.shape[1]:
			self.n_components = x.shape[1]
			print("warning:parameter n_components larger than feature size, defalut set as feture size")
		return self._pca()

	def _pca(self):
		mean_ = np.mean(self.x, axis=0)
		self.feature_x = self.x - mean_
		cov = np.cov(self.feature_x, rowvar=False)
		self.eig_value, self.eig_vector = np.linalg.eig(cov)
		self.reserved_vector = self.eig_vector[:, np.argsort(self.eig_value)][:, -self.n_components:]
		if self.verbose==1:
			self.get_precision()
		return np.dot(self.x, self.reserved_vector)

	def get_direction(self):
		return self.reserved_vector.T

	def get_precision(self):
		self.reserved_value = self.eig_value[np.argsort(self.eig_value)][-self.n_components:]
		precision = np.sum(self.reserved_value) / np.sum(self.eig_value)
		print("info:after pca get {:.2f}% precision".format(precision*100))


if __name__ == '__main__':
	np.random.seed(2)
	a = np.random.random([100,5])
	b = PCA(4, 1).fit(a)
	print(b)