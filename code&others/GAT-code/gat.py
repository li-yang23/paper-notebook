import numpy as np
import tensorflow as tf

from .utils import layers
from base_gattn import BaseGAttN

class GAT(BaseGAttN):
	def inference(self,inputs,nb_classes,nb_nodes,training,attn_drop,ffd_drop,
				bias_mat,hid_units,n_heads,activation=tf.nn.elu,residual=False):
		"""do not know yet
		
		Arguments:
			inputs {[float,float,float]} -- [list describing the batch size, 
																 number of nodes 
															 and ft_size(dont know yet)]
			nb_classes {[type]} -- [description]
			nb_nodes {[type]} -- [description]
			training {[type]} -- [description]
			attn_drop {[type]} -- [description]
			ffd_drop {[type]} -- [description]
			bias_mat {[type]} -- [masked attention, ensure the coefficient only consider neighbor nodes
								  but do not know how it works yet]
			hid_units {list[int]} -- [dimensions of every hidden layer]
			n_heads {list[int]} -- [number of self attention mechanisms in each layer]
		
		Keyword Arguments:
			activation {function} -- [the activate function of each layer] (default: {tf.nn.elu})
			residual {bool} -- [description] (default: {False})
		"""
		attns = []
		for _ in range(n_heads[0]):
			# ? 以这个为例，我感觉应该是添加了一个输入维度是input，输出维度是hid_units[0]
			# ? 还有一堆参数不想写明/不知道含义
			# ? 的单层神经网络，然后一共添加n_heads[0]个，因为原文里面用了multi-head attention机制
			attns.append(layers.attn_head(inputs,bias_mat=bias_mat,
						out_sz=hid_units[0],activation=activation,
						in_drop=ffd_drop,coef_drop=attn_drop,residual=False))
		h_1 = tf.concat(attns,axis=-1)
		for i in range(1,len(hid_units)):
			h_old = h_1
			attns = []
			for _ in range(n_heads[i]):
				attns.append(layers.attn_head(h_1,bias_mat=bias_mat,
							 out_sz=hid_units[i],activation=activation,
							 in_drop=ffd_drop,coef_drop=attn_drop,residual=residual))
			h_1 = tf.concat(attns,axis=-1)
		out = []
		for i in range(n_heads[-1]):
			out.append(layers.attn_head(h_1,bias_mat=bias_mat,
						out_sz=nb_classes,activation=lambda x:x,
						in_drop=ffd_drop,coef_drop=attn_drop,residual=False))
		logits = tf.add_n(out) / n_heads[-1] # 最后的输出求的是均值
		return logits