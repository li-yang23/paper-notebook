import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq,out_sz,bias_mat,activation,in_drop=0.0,coef_drop=0.0,residual=False):
	"""[summary]
	
	Arguments:
		seq {[type]} -- [description]
		out_sz {[type]} -- [description]
		bias_mat {[type]} -- [description]
		activation {[type]} -- [description]
	
	Keyword Arguments:
		in_drop {float} -- [description] (default: {0.0})
		coef_drop {float} -- [description] (default: {0.0})
		residual {bool} -- [description] (default: {False})
	"""
	# TODO: check what all these tensorflow functions do
	with tf.name_scope('my_attn'):
		if in_drop != 0.0:
			seq = tf.nn.dropout(seq, 1.0-in_drop)
			# TODO: check what tf.nn.dropout do
		seq_fts = tf.layers.conv1d(seq,out_sz,1,use_bias=False)
		# TODO: check what conv1d do and return
		# ! important !
 
		# simplest self-attention possible
		f_1 = tf.layers.conv1d(seq_fts,1,1) # f_1 = W * h_i?
		f_2 = tf.layers.conv1d(seq_fts,1,1) # f_2 = W * h_j?
		logits = f_1 + tf.transpose(f_2,[0,2,1]) # 这个执行的是特征拼接？
		coefs = tf.nn.softmax(tf.nn.leaky_relu(logits)+bias_mat)

		if coef_drop != 0.0:
			coefs = tf.nn.drop_out(coefs,1.0-coef_drop)
		if in_drop != 0.0:
			seq_fts = tf.nn.dropout(seq_fts,1.0-in_drop)
		
		# TODO: check functions below
		vals = tf.matmul(coefs,seq_fts)
		ret = tf.contrib.layers.bias_add(vals)

		if residual:
			if seq.shape[-1] != ret.shape[-1]:
				ret = ret + conv1d(seq,ret.shape[-1],1)
			else:
				ret = ret + seq
		
		return activation(ret)
	
	def sp_attn_head():
		pass