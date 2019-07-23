import tensorflow as tf

class BaseGAttN:
	def loss(logits,labels,nb_classes,class_weights):
		sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels,nb_classes),class_weights),axis=-1)
		xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=labels,logits=logits),sample_wts)
		return tf.reduce_mean(xentropy,name='xentropy_mean')
	
	def training(loss,lr,l2_coef):
		# weight decay
		vars = tf.trainable_variables()
		lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
							in ['bias','gamma','b','g','beta']])*l2_coef
		# add_n(): 两个张量的元素和，新的依然是一个张量

		# optimizer
		# 使用亚当优化器
		opt = tf.train.AdamOptimizer(learning_rate=lr)

		# training op
		# 亚当优化器的优化目标是loss和lossL2的和
		train_op = opt.minimize(loss+lossL2)

		return train_op
	
	def preshape(logits,labels,nb_classes):
		new_sh_lab = [-1]
		new_sh_log = [-1,nb_classes]
		log_resh = tf.reshape(logits,new_sh_log)
		lab_resh = tf.reshape(labels,new_sh_lab)
		return log_resh,lab_resh
	
	def confmat(logits,labels):
		preds = tf.argmax(logits,axis=1)
		return tf.confusion_matrix(labels,preds)
	
	def masked_softmax_cross_entropy(logits,labels,mask):
		pass
	
	def masked_sigmoid_cross_entropy(logits,labels,mask):
		pass
	
	def masked_accuracy(logits,labels,mask):
		pass
	
	def micro_f1(logits,labels,mask):
		pass