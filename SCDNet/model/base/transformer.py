import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value=None, mask=None): # query, key, value: (b, ha^2, ch)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # b, h, -1, d_      -1 = ha^2 * ch / d_model = ha^2
             for l, x in zip(self.linears, (query, key))]
        # value: (B, C, HW) -> (head, B, C, HW) -> (B, head, HW, C)
        value = value.repeat(self.h, 1, 1, 1).permute(1, 0, 3, 2).contiguous()

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query, key, value, mask=mask, dropout=self.dropout) # b, h, -1, -1
        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3), query # b, -1, -1

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) # b, h, -1, -1
    if mask is None:
        m = torch.zeros_like(scores)
    else:
        m = torch.zeros_like(scores)
    scores = (scores + m) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1) # b, h, -1, -1
    if dropout is not None:
        scores = dropout(p_attn)
    
    # import pdb; pdb.set_trace()
    return torch.matmul(p_attn, value) # b, h, -1, -1
   
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])