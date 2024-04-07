import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.Wq(x).view(batch_size, -1, self.num_heads, self.depth)
        K = self.Wk(x).view(batch_size, -1, self.num_heads, self.depth)
        V = self.Wv(x).view(batch_size, -1, self.num_heads, self.depth)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.depth)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        #attention = F.relu(scores)
        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        out = self.fc(out)
        return out
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)
        if d_model %2 == 1:
          d_model_1 = d_model + 1
        else:
          d_model_1 = d_model
        pe = torch.zeros(max_len, d_model_1).float()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_1, 2).float() * (-math.log(10000.0) / d_model_1))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        #self.register_buffer('pe', pe)
        self.pe = pe[:,:d_model] #.to(device)
        self.register_buffer('pe_const', self.pe)

    def forward(self, x):
        #print(self.pe.shape)
        #print(x.shape) # 128, 10, 50, 1
        #print(self.pe[:x.size(1), :].shape)
        x = x + self.pe_const[:x.size(1), : ][None, :,:] # 128, 50, 38
        return x


class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AttentionBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        #self.dropout = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm(x)
        x = self.multi_head_attention(x, mask)
        x = self.relu(x) + residual
        residual = x
        x = self.layer_norm2(x)
        x = self.linear(x)
        x += residual
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, input_len, output_dim, output_len, num_hidden_layers, num_heads=2):
        super(Transformer, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_dim = output_dim
        self.output_len = output_len

        #encoding
        #in_size = input_len*input_dim
        #out_size = 1*output_dim
        #decoding
        #in_size = 1*input_dim
        #out_size = output_len*output_dim

        self.in_size = input_len*input_dim
        self.out_size = output_len*output_dim


        self.linear_in = nn.Linear(input_dim, output_dim)
        self.layers = [PositionalEncoding(output_dim)]
        self.layers += [AttentionBlock(output_dim, num_heads) for i in range(num_hidden_layers)]
        self.layers = nn.Sequential(*self.layers)
        self.linear_out = nn.Linear(input_len*output_dim, output_len*output_dim)

        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #print("Transformer input x.shape", x.shape)
        #x = x.reshape(-1, self.input_len, self.input_dim)
        #print("Transformer  x.reshape(-1, self.input_len, self.input_dim)", x.shape)
        x = self.linear_in(x)
        #x = x.reshape(-1, self.input_len, self.output_dim)
        #print("Transformer x.shape self.linear_in(x)", x.shape)
        x = self.layers(x)
        #print("Transformer self.layers(x) ", x.shape)
        x = x.reshape(-1, self.input_len * self.output_dim)
        #x = x.transpose(1,2)
        x = self.linear_out(x)

        x = x.reshape(-1, self.output_len, self.output_dim)
        #x = x.transpose(1,2)
        #print("Transformer self.linear_out(x)", x.shape)
        #x = x.reshape(-1, self.out_size)
        return x


class LSTM(nn.Module):
    def __init__(self, input_dim, input_len, output_dim, output_len, num_hidden_layers=1, hidden_size=128, drop_out=0.05):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_dim = output_dim
        self.output_len = output_len

        #encoding
        #in_size = input_len*input_dim
        #out_size = 1*output_dim
        #decoding
        #in_size = 1*input_dim
        #out_size = output_len*output_dim

        self.in_size = input_len*input_dim
        self.out_size = output_len*output_dim


        self.linear_skip = nn.Linear(self.in_size, self.out_size)

        # set parameters
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers
        self.drop_out = drop_out

        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_dim, self.hidden_size, self.num_layers, dropout=self.drop_out, batch_first=True) # input, hidden, num_layer

        self.dropout = nn.Dropout(self.drop_out)

        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(hidden_size * input_len, self.out_size)
        self.linear_2 = nn.Linear(self.out_size, self.out_size)


        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #print("LSTM input x.shape", x.shape)
        #x = x.reshape(-1, self.input_len, self.input_dim)
        residual = x.reshape(-1, self.in_size)
        #print("LSTM  x.reshape(-1, self.input_len, self.input_dim)", x.shape)

        x, _ = self.lstm(x)

        x = x.reshape(-1, self.hidden_size * self.input_len)
        #print("LSTM x.shape  x = x.reshape(-1, self.hidden_size * self.input_len)", x.shape)
        x = self.relu(self.dropout(self.linear_1(x)))
        x = self.dropout(self.linear_2(x))
        x = F.relu(x + self.linear_skip(residual))

        x = x.reshape(-1, self.output_len, self.output_dim)
        #x = x.transpose(1,2)
        #print("LSTM self.linear_out(x)", x.shape)
        #x = x.reshape(-1, self.out_size)
        return x

