'''
Descripttion:
version:
Date: 2021-06-17 16:10:01
LastEditTime: 2021-06-17 16:51:32
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
import torch.optim as optim
from torch.autograd import Variable
from attention import Multi_Head_SelfAttention
import torch
import math

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input) # XW
        output = torch.spmm(adj, support) # AXW
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nclass, nhid, dropout)
        self.SGCN2 = GCN(nfeat, nclass, nhid, dropout)
        self.CGCN  = GCN(nfeat, nclass, nhid, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid, nclass),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output #, att, emb1, com1, com2, emb2, emb

class JINAN_model(nn.Module):
    """
    multi_slice model.
    """
    def __init__(self, num_head, num_slice, nfeat, nhid, nclass, dropout, degree):
        super(JINAN_model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_head = num_head
        self.num_slice = num_slice
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.degree = degree
        self.model_list = []

        for i in range(self.num_slice):
            self.model_list.append(
                SFGCN(self.nfeat, self.nclass, self.nhid, self.dropout)
            )
        self.attention = Multi_Head_SelfAttention(num_head=self.num_head, num_vocab=self.num_head, input_dim=self.nclass, hidden_dim=self.nclass, out_dim=self.nclass)

    def return_attention_index(self, idx):

        if idx== 0 or idx==1 or idx==2:
            return [0,1,2]
        elif idx < 12*1:
            return [idx-2, idx-1, idx]
        elif idx < 12*2: # 24*2
            return [idx-12, idx-2, idx-1, idx]
        elif idx < 12*7: # 168
            return [idx-12*2, idx-12, idx-2, idx-1, idx]
        elif idx < 12*14: # 336
            return [idx-12*7, idx-12*2, idx-12, idx-2, idx-1, idx]
        else:
            return [idx-12*14, idx-12*7, idx-12*2, idx-12, idx-2, idx-1, idx] # TODO: 12 slice


    def forward(self, x, sadj, fadj, use_relu=True):

        output_list = []
        for i in range(self.num_slice):
            cur_output = self.model_list[i](x, sadj, fadj).unsqueeze(0)
            output_list.append(cur_output)

        output = torch.cat(output_list,dim=0)   # (774,433,128)
        output = output.transpose(0,1)          # (433,774,128)
        attention_output_list = []
        for idx in range(self.num_slice):
            cur_idx = self.return_attention_index(idx)
            if idx==0 or idx==1 or idx==2:
                attention_output = self.attention(output[:,cur_idx,:]).transpose(0,1)[idx] # (433,3,128)->(3,433,128)
                attention_output_list.append(attention_output.unsqueeze(0))
            else:
                attention_output = self.attention(output[:,cur_idx,:]).transpose(0,1)[-1]  # (433,x,128)->(x,433,128)->unsqueeze(0)
                attention_output_list.append(attention_output.unsqueeze(0))
        return torch.cat(attention_output_list,dim=0)