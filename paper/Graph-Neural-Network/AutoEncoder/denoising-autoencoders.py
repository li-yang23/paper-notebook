"""
Denoising Autoencoder 去噪自动编码器
去噪自动编码器是堆叠自动编码器的构建模块。
首先使用输入x并将其映射到隐层表征空间y上
    y = f(x) = s(Wx+b)
然后在将其映射回原空间
    z = g(y) = s(W'y+b')
然后目标就是最小化重建后的z和x的误差（即重建误差）

对于去噪自动编码器，将第一个x通过随机映射破坏成 ~x，
然后计算 y = f(~x) = s(W~x+b)
然后反向映射 z = g(y) = s(W'y+b')
然后重建误差使用z和未破坏的x之间测量
    Loss = sum[k=1 to d](x_k*log(z_k)+(1-x_k)*log(1-z_k))

参考文献：
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
"""
from __future__ import print_function

import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

class dA(object):
    def __init__(self,numpy_rng,theano_rng=None,input=None,n_visible=784,
                    n_hidden=500,W=None,bhid=None,bvis=None):
        """通过指定可见的单元数量，隐藏单元的数量和损坏级别来初始化dA类。
        构造函数还接收输入，权重和偏差的符号变量。
        
        Arguments:
            numpy_rng {numpy.random.RandomState} -- number of random generator used to generate weights
        
        Keyword Arguments:
            theano_rng {[type]} -- theano random generator (default: {None})
            input {[type]} -- a symbolic description of the input or None for standalone dA (default: {None})
            n_visible {int} -- [description] (default: {784})
            n_hidden {int} -- [description] (default: {500})
            W {[type]} -- [description] (default: {None})
            bhid {[type]} -- [description] (default: {None})
            bvis {[type]} -- [description] (default: {None})
        """ 
        self.n_visible = n_visible # 可见的单元数量
        self.n_hidden = n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        if not W:
           initial_W = numpy.asarray(
               numpy_rng.uniform(
                   low = -4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                   high = 4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                   size = (n_visible,n_hidden)
                ),
            dtype=theano.config.floatX
           )
           W = theano.shared(value=initial_W,name='W',borrow=True)
        
        if not bvis: # b for visible,即可见层的偏置
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        if not bhid: # b for hidden, 即隐藏层的偏置
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        self.params = [self.W, self.b, self.b_prime]
    
    def get_corrupted_input(self,input,corruption_level):
        return self.theano_rng.binomial(size=input.shape,n=1,
        p=1-corruption_level,
        dtype=theano.config.floatX) * input

