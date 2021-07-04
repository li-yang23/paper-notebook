class SdA(object):
	"""[Stacked denoising auto-encoder class]
	a stacked denoising autoencoder model is obtained by stacking several
	dAs. The hidden layer of the dA at layer 'i' becomes the input of the
	dA at layer 'i+1'. The first layer dA gets as input the input of the 
	SdA, and the hidden layer of the last dA represents the output.
	Note that after pretraining, the SdA is dealt with as a normal MLP,
	the dAs are only used to initialize the weights.
	"""
	def __init__(
		self,
		numpy_rng,
		theano_rng=None,
		n_ins=784,
		hidden_layers_sizes=[500,500],
		n_outs=10,
		corruption_levels=[0.1,0.1]
	):
		"""this class is made to support a variable number of layers.
		
		Arguments:
			numpy_rng {numpy.random.RandomState} -- [numpy random number generator used to 
													draw initial weights]

		Keyword Arguments:
			theano_rng {theano.tensor.shared_randomstreams.RandomStreams} 
				-- [theano random generator; if None then 
					one is generated based on a seed drawn from 'rng'] (default: {None})
			
			n_ins {int} -- [dimension of the input to the sdA] (default: {784})
			
			hidden_layers_sizes {list[int]} -- [intermediate layers size, 
										   must contain at least one value] (default: {[500,500]})
			
			n_outs {int} -- [dimension of the output of the network] (default: {10})

			corruption_levels {list[float]} 
				-- [amount of corruption to use for each layer,
					dA partially corrupt the input data before training] (default: {[0.1,0.1]})
		"""
		self.sigmoid_layers = []
		self.dA_layers = []
		self.params = []
		self.n_layers = len(hidden_layers_sizes)

		assert self.n_layers > 0
		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2**30))
		# allocate symbolic variables for the data
		self.x = T.matrix('x')
		self.y = T.ivector('y')