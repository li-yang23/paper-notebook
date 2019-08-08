import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self,in_ft,out_ft,act,bias=True):
        """[summary]
        
        Arguments:
            nn {[type]} -- [description]
            in_ft {[type]} -- in_feature,输入的特征的维度，感觉应该是个张量，
                                        shape=(batch,node,feature)
            out_ft {[type]} -- out_feature,输出的特征的维度，感觉应该也是个张量，
                                        shape=(batch,node,out_feature)
            act {[type]} -- [description]
        
        Keyword Arguments:
            bias {bool} -- [description] (default: {True})
        """
        super(GCN,self).__init__()
        self.fc = nn.Linear(in_ft,out_ft,bias=False) # 就是一个简单的全连接层？
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            # 如果有偏置，那就初始化一个和输出维度相同的0张量
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias',None)
        
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    # shape of seq: (batch,nodes,features)
    # 所以seq是每批送进来的节点的特征张量
    def forward(self,seq,adj,sparse=False):
        seq_fts = self.fc(seq)
        # 这个就是将节点特征输入进去，然后得到嵌入的特征向量
        if sparse:
            # 如果是稀疏张量
            out = torch.unsqueeze(torch.spmm(adj,torch.squeeze(seq_fts,0)),0)
            # torch.unsqueeze(a,b)是在a的第b维增加一个维度
            # torch.squeeze(a,b)是将a的第b维去掉
            # torch.spmm(): 即sparse matrix multiplication稀疏矩阵相乘
        else:
            out = torch.bmm(adj,seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)