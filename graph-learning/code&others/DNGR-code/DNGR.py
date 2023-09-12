from RandSurf import RandSurf
from GetPPMIMatrix import GetPPMIMatrix
from GenRep import GenRep

# ! the original algorithm is written by matlab,
# ! there is a chance that this program is wrong,
# ! please right me if you noticed any mistakes, thank you
def DNGR(adjMat,saeInit,opts,nnsize,Kstep,alpha):
	"""[DNGR algorithm]
	Arguments:
		adjMat {[numpy.ndarray]} -- [adjacency matrix of network]
		saeInit {[type]} -- [description]
		opts {[int]} -- [number of layers]
		nnsize {[list(numpy.array)]} -- [architecture of neural network]
		Kstep {[type]} -- [description]
		alpha {[type]} -- [description]
	"""
	# randomly surf to generate K steps Transition Matrix
	Mk = RandSurf(adjMat,Kstep,alpha)

	# get PPMI Matrix
	PPMI = GetPPMIMatrix(Mk)

	# compress the dimension using SDAE
	# TODO: write the stacked denoising autoencoder
	# TODO: write the neural network, 
			# but why use a neural network?
			# hence already have a autoencoder
	sae = saetrain(saeInit,PPMI,opts)
	rep = GenRep(PPMI,sae,nnsize)

	return rep