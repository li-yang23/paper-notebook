# this is the pipeline of deep graph infomax
# https://github.com/PetarV-/DGI/blob/master/layers/gcn.py

import torch
import torch.nn as nn
from gcn import GCN
from readout import AvgReadout
from discriminator import Discriminator

class DGI(nn.Module):
    def __init__(self,n_in,n_h,activation):
        """[summary]
        整个pipline包括一个图卷积网络，一个readout函数和一个分类器（discriminator）？
        
        首先是要利用损坏函数（corruption function）对网络进行负采样
        然后是将输入的图输入图卷积网络得到patch表征（补丁表征？只聚合了网络部分特征的局部表征）
        然后将负例传入图卷积网络得到负例的patch表征
        最后将输入图的patch表征进行结合得到global表征s

        Arguments:
            nn {[type]} -- [description]
            n_in {[type]} -- 输入的数量？啥数量，输入的节点数目？
            n_h {[type]} -- [description]
            activation {[type]} -- 激活函数？？蛤？？
        """
        super(DGI,self).__init__()
        self.gcn = GCN(n_in,n_h,activation) # 图卷积网络得到节点的patch表征
        self.read = AvgReadout() # readout函数将所有的patch表征聚合起来，得到图级别的表征s

        self.sigm = nn.Sigmoid() # 然后进行激活？得到最后的s表征

        self.disc = Discriminator(n_h) # discriminator计算patch表征和整体表征的互信息？作为loss function
    
    def forward(self,seq1,seq2,adj,sparse,msk,samp_bias1,samp_bias2):
        """[summary]
        
        Arguments:
            seq1 {[type]} -- [description]
            seq2 {[type]} -- [description]
            adj {[type]} -- 图的邻接矩阵
            sparse {[type]} -- [description]
            msk {[type]} -- [description]
            samp_bias1 {[type]} -- [description]
            samp_bias2 {[type]} -- [description]
        """
        h_1 = self.gcn(seq1,adj,sparse)
        c = self.read(h_1,msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2,adj,sparse)
        ret = self.disc(c,h_1,h_2,samp_bias1,samp_bias2)

        return ret

    def embed(self,seq,adj,sparse,msk):
        h_1 = self.gcn(seq,adj,sparse)
        c = self.read(h_1,msk)

        return h_1.detach(),c.detach()