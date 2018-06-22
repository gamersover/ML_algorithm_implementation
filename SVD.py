#coding:utf-8
"""
svd算法实现
"""
import numpy as np

def svd(a):

	sigma2_u, u = np.linalg.eig(np.dot(a, a.T))
	sigma2_v, v = np.linalg.eig(np.dot(a.T, a))

	sigma2_u = np.real(sigma2_u)
	sigma2_v = np.real(sigma2_v)
	u = np.real(u)
	v = np.real(v)

	sigma = np.zeros_like(a)
	m = min(a.shape[0], a.shape[1])

	u_sort = u[:, np.argsort(sigma2_u)][:, ::-1]
	v_sort = v[:, np.argsort(sigma2_v)][:, ::-1]

	sigma2 = sigma2_u[np.argsort(sigma2_u)][::-1]
	for i in range(m):
		sigma[i,i] = np.sqrt(sigma2[i])

	u_ = u_sort[:,:]
	for i in range(m):
		u_[:, i] = np.dot(a, v_sort[:, i]) / sigma[i, i]

	v_ = v_sort[:,:]

	return u_, sigma, v_.T


if __name__ == "__main__":
	a = np.random.random([4,4])
	u, sigma, v = svd(a)
	print(a)
	print(u @ sigma @ v)

