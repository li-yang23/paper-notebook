# generate graph representations

def GenRep(R,sae,nnsize):
	"""[summary]

	Arguments:
		R {[type]} -- [description]
		sae {[type]} -- [description]
		nnsize {[list(numpy.array)]} -- [architecture of neural network,
										number of nodes in every layer of the network]
		nnsize is like: [780,600,300,100,10]
	"""
	# l is the number of layers? 
	# the first layer is input layer
	# the last layer is output layer
	# layers between them are all hidden layers 
	# 	with different number of nodes
	l = len(nnsize)

	# create a forward propagation neural network
	# nnFF = nnsetup(nnsize)
	# nnFF.activation_function = 'sigm'

	num_layers = len - 1
	for i in range(1,num_layers+1):
		# nnFF.W{i} = sae.ae{i}.W{1}
		pass
	
	# nnFF.testing = 1
	# nnFF = nnff(nnFF,R,zeros(size(R,1),nnFF.size(end)))
	# rep = nnFF.a{end}
	return rep