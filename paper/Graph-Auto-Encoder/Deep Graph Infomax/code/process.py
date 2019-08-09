import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn

def parse_skipgram(fname):
	"""[summary]
	
	Arguments:
		fname {[type]} -- [description]
	"""
	with open(fname) as f:
		toks = list(f.read().split())
	nb_nodes = int(toks[0])
	nb_features = int(toks[1])
	ret = np.empty((nb_nodes,nb_features))
	it = 2
	for i in range(nb_nodes):
		cur_nd = int(toks[it]) - 1
		it += 1
		for j in range(nb_features):
			cur_ft = float(toks[it])
			ret[cur_nd][j] = cur_ft
			it += 1
	return ret

	def process_tu(data,nb_nodes):
		nb_graphs = len(data)
		ft_size = data.num_features

		features = np.zeros((nb_graphs,nb_nodes,ft_size))
		adjacency = np.zeros((nb_graphs,nb_nodes,nb_nodes))
		labels = np.zeros(nb_graphs)
		sizes = np.zeros(nb_graphs,dtype=np.int32)
		masks = np.zeros((nb_graphs,nb_nodes))

		for g in range(nb_graphs):
			sizes[g] = data[g].x.shape[0]
			features[g,:sizes[g]] = data[g].x
			labels[g] = data[g].y[0]
			masks[g,:sizes[g]] = 1.0
			e_ind = data[g].edge_index
			coo = sp.coo_matrix((np.ones(e_ind.shape[1]),(e_ind[0,:],e_ind[1,:])),shape=(nb_nodes,nb_nodes))
			adjacency[g] = coo.todense()
		return features,adjacency,labels,sizes,masks

def micro_f1(logits,labels):
	# aha! this is the loss function i think
	pass
