import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self,in_ft,out_ft,act,bias=True):
        """[summary]
        
        Arguments:
            nn {[type]} -- [description]
            in_ft {[type]} -- in_feature,输入的特征的维度
            out_ft {[type]} -- out_feature,输出的特征的维度
            act {[type]} -- [description]
        
        Keyword Arguments:
            bias {bool} -- [description] (default: {True})
        """
        if bias:
            # 如果有偏置，那就
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
    
    def forward(self,seq,adj,sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj,torch.squeeze(seq_fts,0)),0)
        else:
            out = torch.bmm(adj,seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)