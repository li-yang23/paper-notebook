from ScaleSimMat import ScaleSimMat
import numpy as np

def GetPPMIMatrix(M):
	"""[get PPMI matrix of network, contains only the 
		positive transition probability from between nodes]
	
	Arguments:
		M {[numpy.array]} -- [trainsition matrix of network]
	"""
	M = ScaleSimMat(M)
	
	p,q = M.shape
	assert p==q,"M must be a square matrix, sorry"

	col = np.sum(M,axis=0)
	row = np.sum(M,axis=1)

	D = sum(col)
	# PPMI = log(D * M ./ (row*col))
	PPMI = np.log(np.dot(D,M)/np.dot(row,col))
	# * change negative elements in PPMI to 0
	# PPMI(PPMI<0)=0
	PPMI = np.maximum(PPMI,0)

	# * change nan element into 0, because it means that these 
	# * two nodes basically have no chance to transfer to each other
	IdxNan = PPMI
	IdxNan[np.isnan(IdxNan)] = 1
	IdxNan[~np.isnan(IdxNan)] = 0
	PPMI[IdxNan] = 0
	return PPMI