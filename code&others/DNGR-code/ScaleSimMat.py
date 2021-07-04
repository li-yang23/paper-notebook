import numpy as np

def ScaleSimMat(W):
	"""[summary]
	normalize the transition matrix
	Arguments:
		W {[numpy.ndarray]} -- [transition matrix of network]
	"""
	# * make diagonal elements 0 because 
	# * we do not concern restart probabilities here
	# W = W - diag(diag(W))
	W = W - np.diag(np.diag(W))
	D = np.diag(np.sum(W,axis=0),0)
	# W = D^(-1)*W
	
	W = np.linalg.inv(D)*W
	return W