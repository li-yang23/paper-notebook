import numpy as np
import math

def lfm(a,k):
	"""latent factor model
	
	Arguments:
		a {[numpy.ndarray]} -- [the rating matrix that needs to factorize]
		k {[int]} -- [number of latent variables]
	
	A = U*V   U->m*k; V->k*n
	"""
	assert type(a) == np.ndarray
	m,n = a.shape
	alpha = 0.01
	lambda_ = 0.01
	u = np.random.randn(m,k)
	v = np.random.randn(k,n)
	for t in range(1000):
		for i in range(m):
			for j in range(n):
				if math.fabs(a[i][j]) > 1e-4:
					err = a[i][j] - np.dot(u[i],v[:,j])
					for r in range(k):
						gu = err * v[r][j] - lambda_ * u[i][r]
						gv = err * u[i][r] - lambda_ * v[r][j]
						u[i][r] += alpha * gu
						v[r][j] += alpha * gv
	return u,v