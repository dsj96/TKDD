
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import softmax
import torch.optim as optim
from torch.autograd import Variable

import math
import numpy as np

from utils import write_pkl, read_pkl

class SelfAttention(nn.Module):

    def __init__(self, num_vocab, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.num_vocab  = num_vocab
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.query_W = nn.Linear(input_dim, hidden_dim)
        self.key_W   = nn.Linear(input_dim, hidden_dim)
        self.value_W = nn.Linear(input_dim, hidden_dim)
        self.position_encoder = PositionalEncoder(d_model=hidden_dim)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.query_W.weight.size(1))
        self.query_W.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.query_W.weight.size(1))
        self.key_W.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.query_W.weight.size(1))
        self.key_W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input_data):
        query_M = self.query_W(input_data)
        key_M   = self.key_W(input_data)
        value_M = self.value_W(input_data)
        attn_scores = query_M @ key_M.transpose(-1,-2)
        attn_scores = attn_scores / math.sqrt(self.hidden_dim)
        attn_scores_softmax = softmax(attn_scores, dim=-1)

        outputs = self.position_encoder(attn_scores_softmax.bmm(value_M))
        return outputs



class Multi_Head_SelfAttention(nn.Module):

    def __init__(self, num_head, num_vocab, input_dim, hidden_dim, out_dim):
        super(Multi_Head_SelfAttention, self).__init__()
        self.num_head   = num_head
        self.num_vocab  = num_vocab
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.fn = nn.Linear(num_head*hidden_dim, out_dim)


        self.at_block_list = []
        for i in range(self.num_head):
            self.at_block_list.append(SelfAttention(self.num_vocab, self.input_dim, self.hidden_dim))


    def forward(self, input_data):
        output_list = []
        for i in range(self.num_head):
            cur_output = self.at_block_list[i](input_data)
            output_list.append(cur_output)
        output_M = torch.cat(output_list,dim=-1)
        outputs = self.fn(output_M)
        return outputs

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 30):
        super(PositionalEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.max_seq_len = max_seq_len


        self.pe = torch.zeros((self.max_seq_len, self.d_model), requires_grad=False)
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                self.pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                if i+1< self.d_model:
                    self.pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len,:self.d_model]
        return x


if __name__ == "__main__":
    '''self-attention'''
    input_data = torch.tensor(np.arange(24.).reshape(2,3,4),dtype=torch.float32)
    attention = Multi_Head_SelfAttention(num_head=3,num_vocab=3,input_dim=4,hidden_dim=5,out_dim=5)
    output_data = attention(input_data)
    print("output_data: \n",output_data)

    '''position encoder'''
    # input_data = torch.tensor([[[1,0,1,0],[0,2,0,2],[1,1,1,1]], [[1,0,1,0],[0,2,0,2],[1,1,1,1]] ],dtype=torch.float32)
    # print("input_data:\n",input_data)
    # positional_encoder = PositionalEncoder(d_model=4) # 该参数代表输出的嵌入维度，因为需要和输入的
    # output_data = positional_encoder(input_data)
    # print("output_data:\n", output_data)