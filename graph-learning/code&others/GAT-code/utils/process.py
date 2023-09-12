import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def adj_to_bias(adj,size,nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[h] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g],(adj[g]+np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def parse_index_file(dataset_str):
    pass

def load_data(dataset_str):
    names = ['x','y''tx','ty','allx','ally','graph']
    objects = []
    for i in range(len(names)):
        with open('data/ind.{}.{}'.format(dataset_str,names[i]),'rb') as f:
            if sys.version_info > (3,0):
                objects.append(pkl.load(f,encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x,t,tx,ty,allx,ally,graph = tuple(objects)
    test_idx_reorder = parse_index_file('data/ind.{}.test.index'.format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # 将孤立节点作为零向量嵌入合适位置
        test_idx_range_full = range(min(test_idx_reorder),max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full),x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range),:] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full),y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range),:] = ty
        ty = ty_extended
    features = sp.vstack((allx,tx)).tolil()
