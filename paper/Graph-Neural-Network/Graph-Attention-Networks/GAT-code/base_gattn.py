import tensorflow as tf

class BaseGAttN:
	def loss(self,logits,labels,nb_classes,class_weights):
		sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels,nb_classes),class_weights),axis=-1)
		xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=labels,logits=logits),sample_wts)
		return tf.reduce_mean(xentropy,name='xentropy_mean')
	
	def training(self,loss,lr,l2_coef):
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
	
	def preshape(self,logits,labels,nb_classes):
		new_sh_lab = [-1]
		new_sh_log = [-1,nb_classes]
		log_resh = tf.reshape(logits,new_sh_log)
		lab_resh = tf.reshape(labels,new_sh_lab)
		return log_resh,lab_resh
	
	def confmat(self,logits,labels):
		preds = tf.argmax(logits,axis=1)
		return tf.confusion_matrix(labels,preds)
	
	def masked_softmax_cross_entropy(self,logits,labels,mask):
		loss = tf.nn.sofrmax_cross_entropy_with_logits(logits=logits,labels=labels)
		mask = tf.cast(mask,dtype=tf.float32)
		mask /= tf.reduce_mean(mask)
		loss *= mask
		return tf.reduce_mean(loss)
	
	def masked_sigmoid_cross_entropy(self,logits,labels,mask):
		labels = tf.cast(labels,dtype=tf.float32)
		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)
		loss = tf.reduce_mean(loss,axis=1)
		mask = tf.cast(mask,dtype=tf.float32)
		mask /= tf.reduce_mean(mask)
		loss *= mask
		return tf.reduce_mean(loss)
	
	def masked_accuracy(self,logits,labels,mask):
		correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
		accuracy_all = tf.cast(correct_prediction,tf.float32)
		mask = tf.cast(mask,dtype=tf.float32)
		mask /= tf.reduce_mean(mask)
		accuracy_all *= mask
		return tf.reduce_mean(accuracy_all)
	
	def micro_f1(self,logits,labels,mask):
		pass