"""AutoEncoder自编码器，可以利用自身的高阶特征编码自己
本质也是一种神经网络，他的输入和输出是一致的，他接著稀疏编码的思想，使用稀疏的一些高阶特征重新组合来重构自己。
第一，期望输入和输出一致
第二，希望使用高阶特征来重构自己，而不只是复制像素点，因为要是一一对应的那还练个锤子

如果限制中间隐含层节点的数量，让隐含层节点数量小于输入输出节点的数目，这相当于一个将为的过程。
这样已经不可能出现复制所有节点的情况，因为中间节点数小于输入节点数，所以只能学习数据中最重要的特征复原，将不太相关的内容去除。
"""
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
    自编码器中会使用一种参数初始化方法xavier initialization，
    它的特点是会根据某一层网络的输入，输出节点数量自动调整最合适的分布。
    如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用，
    但如果权重初始化得太大，那信号将在每层间传递时逐渐放大并导致发散和失效。
    而Xaiver初始化器做的事情就是让权重被初始化得不大不小，正好合适。
    即让权重满足0均值，同时方差为2／（n（in）+n(out)），分布可以用均匀分布或者高斯分布。
    下面fan_in是输入节点的数量，fan_out是输出节点的数量。
'''
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in+fan_out))
    high = constant * np.sqrt(6.0 / (fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                optimizer=tf.train.AdamOptimizer(),scale=0.1):
        """构造函数
        
        Arguments:
            n_input {tensorflow.scaler} -- 输入变量数
            n_hidden {tensorflow.scaler} -- 隐含层节点数
        
        Keyword Arguments:
            transfer_function {function} -- 隐含层激活函数 (default: {tf.nn.softplus})
            optimizer {tensorflow.train.xx} -- 优化器 (default: {tf.train.AdamOptimizer()})
            scale {float} -- 高斯噪声系数 (default: {0.1})
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()  # 使用initialize_weights进行参数初始化
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input)), # 这里是给输入加了个噪声，
                                                    self.weights['w1']), # tf.matmul函数是矩阵的点乘，就是元素乘法
                                                    self.weights['b1']))
        # 这里是只有一层输入层，一层隐含层和一层重建层
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        # tensorflow.reduce_sum 是进行压缩求和的的函数
        # tensorflow.subtract 是进行元素减法的函数
        # tensorflow.pow 是进行元素乘方的函数
        # 所以损失函数是重建的x和原来的x的元素差的乘方的和，即（basically）欧几里得距离
        self.optimizer = optimizer.minimize(self.cost)
        # 然后优化器的目标是最小化重建误差
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def _initialize_weights(self):
        """参数初始化函数
        """
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights
    
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),
                                  feed_dict={self.x:X,self.scale:self.training_scale})
        return cost
    
    def calc_total_cost(self,X):
        """计算损失，可以用来算测试集上的效果
        
        Arguments:
            X {tensorflow.tensor} -- 输入数据，可以是测试集
        """
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
    
    def transform(self,X):
        """将输入通过隐含层得到中间结果
        """
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

    def generate(self,hidden=None):
        """将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
        
        Keyword Arguments:
            hidden {[type]} -- 隐藏层节点数 (default: {None})
        """
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    

def get_random_block_from_data(data,batch_size):
    """从数据里面随机取一块指定规模的数据
    
    Arguments:
        data {[type]} -- 全体数据
        batch_size {[type]} -- 要取的数据规模
    """
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

writer = tf.summary.FileWriter(logdir='logs',graph=autoencoder.sess.graph)
writer.close()

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xz = get_random_block_from_data(X_train,batch_size)
        cost = autoencoder.partial_fit(batch_xz)
        avg_cost += cost / n_samples*batch_size
    if epoch % display_step == 0:
        print('Epoch:','%04d' % (epoch+1),'cost=','{:.9f}'.format(avg_cost))
print('Total cost:' + str(autoencoder.calc_total_cost(X_test)))
