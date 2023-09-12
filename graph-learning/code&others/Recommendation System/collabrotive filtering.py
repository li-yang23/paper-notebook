import numpy as np
import pandas as pd
from pandas import DataFrame

def euclidea_sim(x,y):
	"""euclidea distance measure
	sim = |x-y|
	Arguments:
		x {[numpy.ndarray]} -- [first veator]
		y {[numpy.ndarray]} -- [second vector]
	"""
	assert len(x) == len(y)
	dis = np.linalg.norm(x-y)
	sim = 1/(1+dis) # * 1+dis: just in case that x==y thus dis=0
	return sim

def jaccard_sim(x,y):
	"""jaccard similarity measure
	sim=|x and y|/|x or y|
	Arguments:
		x {[numpy.ndarray]} -- [first vector]
		y {[numpy.ndarray]} -- [second vector]
	"""
	assert len(x) == len(y)
	x,y = np.array(x).astype(bool),np.array(y).astype(bool)
	return sum(x*y)/sum(x+y)

def cosine_sim(x,y):
	"""cosine similarity measure
	sim=x*y/(|x|*|y|)
	Arguments:
		x {[numpy.ndarray]} -- [first vector]
		y {[numpy.ndarray]} -- [second vector]
	"""
	assert len(x) == len(y)
	sum_x_y = np.dot(x,y)
	return sum_x_y/np.linalg.norm(x)/np.linalg.norm(y) # interesting

self_file = ""
sell_record = pd.read_csv(self_file,sep=',',header=0,encoding='gbk')
sell_pivot = sell_record.pivot_table(values='购买数量',index='客户ID',columns='产品ID',aggfunc=sum,fill_value=0)

def sim_mat(sell_group,sim=euclidea_sim):
	"""calculate similarity matrix
	
	Arguments:
		sell_group {pandas.DataFrame} -- [description]
	
	Keyword Arguments:
		sim {[function]} -- [similarity measure function] (default: {euclidea_sim})
	"""
	sim_matrix = np.zeros((sell_group.shape[0],sell_group.shape[0]),dtype=float)
	sim_matrix = DataFrame(sim_matrix,index=sell_group.index,columns=sell_group.index)

	for index in sell_group.index:
		for column in sell_group.index:
			sim_matrix.ix[index,column] = sim(sell_group.ix[index,:],sell_group.ix[column,:])
	return sim_matrix

def recommendation(sim_mat,customer,n_sim_customer,n_product,sell_record):
	"""collaboration filtering recommendation
	
	Arguments:
		sim_mat {[np.ndarray]} -- [similarity matrix]
		customer {[type]} -- [user who needs recommendation]
		n_sim_customer {[type]} -- [number of similar customers]
		n_product {[type]} -- [number of products recommended]
		sell_record {[type]} -- [list of product customer buyed, every data is a row, 
								every column is the number of item the customer purchased]
	"""
	try:
		k_similar = sim_mat.sort_values(customer,axis=0,ascending=False)[:n_sim_customer]
	except:
		print('the custmer did not buy anything yet')
		return None
	recom_product = sell_record.ix[k_similar.index,:].astype(bool).sum(axis=0)
	recom_product = recom_product[recom_product>0].sort_values(axis=0,ascending=False).index
	count_ = 0
	recom_list = []
	for i in recom_product:
		if sell_record[i][customer] > 0:
			continue
		else:
			recom_list.append(i)
			count_ += 1
		if count_ >= n_product:
			return recom_list
	return recom_list