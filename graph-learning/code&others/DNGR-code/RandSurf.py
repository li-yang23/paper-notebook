from ScaleSimMat import ScaleSimMat
import numpy as np

# randsurf algorithm to generrate k-step transition matrix
def RandSurf(A,max_step,alpha):
	"""[random surf algorithm]
	
	Arguments:
		A {[numpy.ndarray]} -- [transition matrix of network]
		max_step {[type]} -- [description]
		alpha {[gloat?]} -- [description]
	"""
	num_nodes = len(A)
	A = ScaleSimMat(A)

	# P0 = eye(num_nodes,num_nodes)
	# P0 is an identity matrix of n*n
	P0 = np.eye(num_nodes) # or P0 = np.eye(num_nodes,num_nodes)
	P = P0
	# M = zeros(num_nodes,num_nodes)
	M = np.zeros((num_nodes,num_nodes))

	for i in range(1,max_step+1):
		# P = alpha*P*A+(1-alpha)*P0
		P = np.dot(P,A)*alpha+(1-alpha)*P0
		M = M + P
	
	return M