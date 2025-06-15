
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class Kernel(nn.Module):
    def __init__(self, input_dim, input_len, 
                 output_dim, output_len, **kwargs):
        super(Kernel, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_dim = output_dim
        self.output_len = output_len 

        self.is_in_encoder = False #input_len >= output_len
        self.is_in_decoder = False #input_len >= output_len
        
        # Update attributes with kwargs
        self._update_params(**kwargs)

    def _update_params(self, **kwargs):
        """
        Standard parameter update function using **kwargs
        Updates object attributes with provided keyword arguments
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

class KernelWrapper(nn.Module): # wrapper with rotation and projection matrix
    def __init__(self, kernel, input_dim, input_len,
                 output_dim=1, output_len=1,
                 **kwargs):
        super(KernelWrapper, self).__init__()

        # kernel : kernel(input_dim, input_len, output_dim, output_len)
        assert (issubclass(kernel, Kernel) or issubclass(kernel, nn.Module))

        if isinstance(kernel, nn.Module):
          print(f"kernel {kernel} heiritated nn.Module may not adapt.")

        self.input_dim, self.input_len, self.output_dim, self.output_len = \
                        input_dim, input_len, output_dim, output_len
                        
        self.verbose = kwargs.get('verbose', False)
        self.unet_skip_concat = kwargs.get("unet_skip_concat", False)
        self.unet_skip = kwargs.get("unet_skip", False)
        self.transpose = kwargs.get("transpose", False)
        self.kernel_hidden_layer = kwargs.get("kernel_hidden_layer", 1)

        self.hidden_size_list = []

        if issubclass(kernel, Kernel):
            if self.verbose:
              print("kernel ", kernel, "is a Kernel")
            self.kernel = kernel(input_dim, input_len, output_dim, output_len, **kwargs)
        else:
            assert False, f"kernel {kernel} is not recognized"

        self._unet_skip_output = None
        self._unet_skip_input = None


    def f(self, x):
        if self.verbose :
          print("---KernelWrapper.f(x) Input x.shape: ", x.shape)
        x = self.kernel(x)
        if self.verbose :
          print("---KernelWrapper.f(x) Output x.shape: ", x.shape)
        return x

    def forward(self, x, train=False):
        if self.verbose :
          print("---KernelWrapper.forward(x) Input x.shape:", x.shape)
          print("---train:", train)
          if self._unet_skip_input is not None:
              print("---_unet_skip_input.shape", self._unet_skip_input.shape)
          else:
              print("---_unet_skip_input", self._unet_skip_input)

        if self.transpose and self._unet_skip_input is not None:
            if self.verbose :
                print("self.transpose and self._unet_skip_input")
                print("--x.shape", x.shape)
            if np.prod(x.shape) == np.prod(self._unet_skip_input.shape):
                if self.verbose :
                    print("self.unet_skip_concat, x.shape", self.unet_skip_concat, x.shape)
                if self.unet_skip_concat:
                    x = torch.cat([x, self._unet_skip_input.reshape(x.shape)], dim=-1)
                    #print( "after, ", self.unet_skip_concat, x.shape)
                else:
                    x = x + self._unet_skip_input.reshape(x.shape)
                #print("_unet_skip_input.shape", self._unet_skip_input.shape)
            #x[len(self._unet_skip_input):] = x[len(self._unet_skip_input):] + self._unet_skip_input
        #x = x.transpose(1, 2)   # # (batch, d_model , lag) to (batch, lag, d_model)
        else:
            pass
        if self.verbose :
            print("reshape - > x.shape", x.shape)
        x = x.reshape(-1, self.input_len, self.input_dim)

        if self.verbose :
            print("after reshape - > x.shape", x.shape)

        x = self.f(x)

        if self.verbose :
            print("after x = self.f(x) - > x.shape", x.shape)


        assert x.shape[1] == self.output_len and x.shape[2] == self.output_dim

        if not self.transpose:
          self._unet_skip_output = x
        return x

class Linear(Kernel):
    def __init__(self, input_dim, input_len, 
                 output_dim, output_len, **kwargs):
        super(Linear, self).__init__(input_dim, input_len, 
                 output_dim, output_len, **kwargs)

        # declare parameters with defaults
        self.activation = kwargs.get('activation', 'tanh')
        self.drop_out_p = kwargs.get('drop_out_p', 0.01)
        self.kernel_hidden_layer = kwargs.get('kernel_hidden_layer', 0)

        # compute input and output size
        self.in_size = input_len*input_dim
        self.out_size = output_len*output_dim

        # prepare layers
        self.layers = []

        # check in encoder or decoder 
        self.is_in_encoder = (self.input_len >= self.output_len)
        self.is_in_decoder = not self.is_in_encoder

        # in encoder
        if self.is_in_encoder:
          gap = int((self.in_size - self.out_size) / (self.kernel_hidden_layer + 1))
          self.hidden_size_list = [self.in_size - i * gap for i in range(1, self.kernel_hidden_layer + 1)]
        
        # in decoder
        else:
          gap = int((self.out_size - self.in_size) / (self.kernel_hidden_layer + 1))
          self.hidden_size_list = [self.in_size + i * gap for i in range(1, self.kernel_hidden_layer + 1)]
        # add linear layers
        for i in range(self.kernel_hidden_layer):
            self.layers.append(nn.Linear(self.in_size, self.hidden_size_list[i]))

            if self.activation.lower() == "relu":
                self.layers.append(nn.ReLU())
            elif self.activation.lower() == "tanh": 
                self.layers.append(nn.Tanh())

            self.layers.append(nn.Dropout(self.drop_out_p))
            self.in_size = self.hidden_size_list[i]

        self.layers.append(nn.Linear(self.in_size, self.out_size))

        self.layers = nn.Sequential(* self.layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_len * self.input_dim)
        #print("x.shape,", x.shape)
        x = self.layers(x)
        x = x.reshape(-1, self.output_len, self.output_dim)
        #print("x.shape,", x.shape)
        return x

class LSTM(Kernel):
    def __init__(self, input_dim, input_len, 
                 output_dim, output_len, **kwargs):
        super(LSTM, self).__init__(input_dim, input_len, 
                 output_dim, output_len, **kwargs)

        # declare parameters with defaults
        self.drop_out_p = kwargs.get('drop_out_p', 0.01)
        self.kernel_hidden_layer = kwargs.get('kernel_hidden_layer', 1)

        self.lstm_dim = max(input_dim, output_dim)
        self.lstm_len = max(input_len, output_len)

        # compute input and output size
        self.in_size = input_len*input_dim
        self.out_size = output_len*output_dim
        self.lstm_size = self.lstm_dim*self.lstm_len

        # prepare layers
        self.layers = []

        # check in encoder or decoder 
        self.is_in_encoder = (self.input_len >= self.output_len)
        self.is_in_decoder = not self.is_in_encoder

        # Define the LSTM and Linear layers
        self.linear_projection_in = nn.Linear(self.in_size, self.lstm_size)
        self.linear_projection_out = nn.Linear(self.lstm_size, self.out_size)

        self.lstm = nn.LSTM(self.lstm_dim, self.lstm_dim, 
                            self.kernel_hidden_layer, dropout=self.drop_out_p, 
                            batch_first=True)

    def forward(self, x):
        """
        Forward pass for LSTM. If we are in encoder mode, process the input sequence through LSTM
        and use the last hidden state to compute the final output using the linear layer.
        """
        #print(x.shape)
        x = x.reshape(-1, self.in_size)
        #print(x.shape)
        x = self.linear_projection_in(x)
        #print(x.shape)
        x = x.reshape(-1, self.lstm_len, self.lstm_dim)
        #print(x.shape)

        # Pass through LSTM
        x, (h_n, c_n) = self.lstm(x)  # x, lstm_out contains all hidden states, h_n is the last hidden state

        # Use the last hidden state (h_n) for linear transformation
        #print(x.shape, h_n.shape, c_n.shape)
        # Apply linear transformation
        x = x.reshape(-1, self.lstm_size)
        x = self.linear_projection_out(x)
        x = x.reshape(-1, self.output_len, self.output_dim)
        #print(x.shape)
        return x
    
class RNN(Kernel):
    def __init__(self, input_dim, input_len, 
                 output_dim, output_len, **kwargs):
        super(RNN, self).__init__(input_dim, input_len, 
                 output_dim, output_len, **kwargs)
        
        # declare parameters with defaults
        self.drop_out_p = kwargs.get('drop_out_p', 0.05)
        self.kernel_hidden_layer = kwargs.get('kernel_hidden_layer', 1)
        

        self.rnn_dim = max(input_dim, output_dim)
        self.rnn_len = max(input_len, output_len)

        # compute input and output size
        self.in_size = input_len*input_dim
        self.out_size = output_len*output_dim
        self.rnn_size = self.rnn_dim*self.rnn_len

        # prepare layers
        self.layers = []

        # check in encoder or decoder 
        self.is_in_encoder = (self.input_len >= self.output_len)
        self.is_in_decoder = not self.is_in_encoder

        # Define the LSTM and Linear layers
        self.linear_projection_in = nn.Linear(self.in_size, self.rnn_size)
        self.linear_projection_out = nn.Linear(self.rnn_size, self.out_size)

        self.lstm = nn.RNN(self.rnn_dim, self.rnn_dim, 
                            self.kernel_hidden_layer, dropout=self.drop_out_p, 
                            batch_first=True)

    def forward(self, x):
        """
        Forward pass for LSTM. If we are in encoder mode, process the input sequence through LSTM
        and use the last hidden state to compute the final output using the linear layer.
        """
        #print(x.shape)
        x = x.reshape(-1, self.in_size)
        #print(x.shape)
        x = self.linear_projection_in(x)
        #print(x.shape)
        x = x.reshape(-1, self.rnn_len, self.rnn_dim)
        #print(x.shape)

        # Pass through LSTM
        x, _ = self.lstm(x)  # x, rnn_out contains all hidden states, h_n is the last hidden state

        # Use the last hidden state (h_n) for linear transformation
        #print(x.shape, h_n.shape, c_n.shape)
        # Apply linear transformation
        x = x.reshape(-1, self.rnn_size)
        x = self.linear_projection_out(x)
        x = x.reshape(-1, self.output_len, self.output_dim)
        #print(x.shape)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // self.num_heads

        # Linear layers for queries, keys, and values
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Output linear layer
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Linear projections for Q, K, V
        Q = self.Wq(x).view(batch_size, -1, self.num_heads, self.depth)
        K = self.Wk(x).view(batch_size, -1, self.num_heads, self.depth)
        V = self.Wv(x).view(batch_size, -1, self.num_heads, self.depth)

        # Permute to bring num_heads dimension to second position
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.depth)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

        # Weighted sum of value vectors
        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        # Final linear transformation
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        if d_model % 2 == 1:
            d_model_1 = d_model + 1
        else:
            d_model_1 = d_model

        pe = torch.zeros(max_len, d_model_1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_1, 2).float() * (-math.log(10000.0) / d_model_1))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_const', pe[:, :d_model])

    def forward(self, x):
        x = x + 0.1 * self.pe_const[:x.size(1), :].unsqueeze(0)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AttentionBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.relu = nn.GELU() #nn.LeakyReLU()
        self.linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # First residual connection
        residual = x

        x = self.multi_head_attention(x, mask)
        x = self.relu(x) + residual
        x = self.layer_norm(x)

        # Second residual connection
        #residual = x
        #x = self.linear(x)
        #x = x + residual

        return x

class Transformer(Kernel):
    def __init__(self, input_dim, input_len,
                 output_dim, output_len, **kwargs):
        super(Transformer, self).__init__(input_dim, input_len,
                 output_dim, output_len)
        # declear parameters
        
        self.drop_out_p = kwargs.get('drop_out_p', 0.05)
        self.kernel_hidden_layer = kwargs.get('kernel_hidden_layer', 1)
        self.num_heads = kwargs.get('num_heads', 2)


        self.attention_dim = max(input_dim, output_dim)
        self.attention_len = max(input_len, output_len)

        # compute input and output size
        self.in_size = input_len*input_dim
        self.out_size = output_len*output_dim
        self.attention_size = self.attention_dim*self.attention_len

        # prepare layers
        self.layers = []

        # check in encoder or decoder
        self.is_in_encoder = (self.input_len >= self.output_len)
        self.is_in_decoder = not self.is_in_encoder

        # Define the LSTM and Linear layers
        self.linear_projection_in = nn.Linear(self.in_size, self.attention_size)
        self.linear_projection_out = nn.Linear(self.attention_size, self.out_size)
        self.attention = [] # [PositionalEncoding(d_model=self.attention_dim)]
        self.attention += [AttentionBlock(d_model=self.attention_dim,
                                num_heads=self.num_heads) for i in range(self.kernel_hidden_layer)]
        self.attention = nn.Sequential(*self.attention)

    def forward(self, x):
        """
        Forward pass for LSTM. If we are in encoder mode, process the input sequence through LSTM
        and use the last hidden state to compute the final output using the linear layer.
        """
        #print(x.shape)
        x = x.reshape(-1, self.in_size)
        #print(x.shape)
        x = self.linear_projection_in(x)

        # Pass through LSTM
        x = x.reshape(-1, self.attention_len, self.attention_dim)
        x = self.attention(x)  # x, attention_out contains all hidden states, h_n is the last hidden state

        # Use the last hidden state (h_n) for linear transformation
        #print(x.shape, h_n.shape, c_n.shape)
        # Apply linear transformation
        x = x.reshape(-1, self.attention_size)
        x = self.linear_projection_out(x)
        x = x.reshape(-1, self.output_len, self.output_dim)
        #print(x.shape)
        return x



class KUNetEncoder(nn.Module):
    def __init__(self, input_dim=128, input_len=4,
                 n_width=[1], n_height=[4, 4],
                 output_dim=128, output_len=1,
                 hidden_dim_list=[128]*3,
                 kernel_list=[Linear]*3, kernel_hidden_layer_list=[1]*3,
                 **kwargs):
        super(KUNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.n_width = n_width
        self.n_height = n_height
        self.output_dim = output_dim
        self.output_len = output_len
        self.hidden_dim_list = hidden_dim_list
        self.kernel_list = kernel_list
        self.kernel_hidden_layer_list = kernel_hidden_layer_list

        self.verbose = kwargs.get('verbose', False)
        
        assert isinstance(n_width, list)
        assert isinstance(n_height, list)
        assert isinstance(kernel_list, list)
        assert isinstance(hidden_dim_list, list)
        assert isinstance(kernel_hidden_layer_list, list)

        # Create lag_list for  Optic Nerve transformer model
        # lag_list = [lag, n_height_1, ...,n_height_n, n_width_1, n_width_n]
        self.lag_list = [input_len]
        if not(len(self.n_height) == 1 and self.n_height[0] ==1):
            self.lag_list =  self.lag_list + list(reversed(self.n_height))
        if not(len(self.n_width) == 1 and  self.n_width[0] ==1):
            self.lag_list =  self.lag_list + list(reversed(self.n_width))
            
        if (len(self.n_width) == 1 or  self.n_width[0] == 1) and \
           (len(self.n_height) == 1 or self.n_height[0] == 1) :
            self.lag_list =  self.lag_list + list(reversed(self.n_width))

        if self.verbose:
            print("self.lag_list", self.lag_list)
            print("hidden_dim_list", hidden_dim_list)

        # declear model
        kernel_hidden_layer = kernel_hidden_layer_list[0]
        kernel = kernel_list[0]
        hidden_dim = hidden_dim_list[0]

        self.layers = [KernelWrapper(kernel,
                                    input_dim=input_dim, input_len=self.lag_list[0],
                                    output_dim=hidden_dim, output_len=1,
                                    **{"kernel_hidden_layer" : kernel_hidden_layer}, **kwargs)]
        self.layers = self.layers + [KernelWrapper(kernel_list[i],
                                                    input_dim=hidden_dim_list[i-1], input_len=l,
                                                    output_dim=hidden_dim_list[i], output_len=1,
                                                    **{"kernel_hidden_layer" : kernel_hidden_layer_list[i]}, **kwargs) \
                                                        for i, l in enumerate(self.lag_list[1:-1], 1)]

        kernel = kernel_list[-1]
        kernel_hidden_layer = kernel_hidden_layer_list[-1]
        hidden_dim = hidden_dim_list[-1]
        self.layers.append(KernelWrapper(kernel,
                            input_dim=hidden_dim, input_len=self.lag_list[-1],
                            output_dim=output_dim, output_len=output_len,
                            **{"kernel_hidden_layer" : kernel_hidden_layer}, **kwargs))

        self.layers = nn.Sequential(* self.layers)

        for i, f in enumerate(self.layers):
          if i+1 < len(self.lag_list):
           f.next_layer_lag = self.lag_list[i+1]
           f.next_d_model = hidden_dim_list[i]
          else:
           f.next_d_model = self.output_dim


    def forward(self, x):
        """
        # reshape
        shape : (batch, [height]*lag, [width]*d_model)
        shape : (batch, [height], lag, [width], d_model)
        shape : (batch, [width], [height], lag, d_model)

        # layer 1 : process height
        1st step:  x -> model(x)
                (batch * [width] * [height] , lag, d_model) -> (batch * [width] * [height], 1, d_model)
                output shape : (batch * [width], lag=height, d_model)

        # layer 2 : process width
        2st step:  x -> model(x)
                (batch * [width], lag=1, d_model)) -> (batch, lag=width, d_model)
                output shape : (batch, lag=1, d_model))

        # layer 3 : process output
        3st step:  x -> model(x)
              (batch, lag=1, d_model)) -> (batch, out_size)
        """
        x_shape = x.shape # (batch, height*lag, width*d_model)
        if self.verbose:
          print("-KUN-Encoder.forward(x) Input x.shape: ", x.shape)

        # layer 1 : process height
        x = x.reshape((-1,)+ (np.prod(self.n_height),) +(self.input_len,) + (np.prod(self.n_width),) + (1,) + (self.input_dim,))
        # (batch, [height], lag, [width], d_model)
        if self.verbose:
          print("-KUN-Encoder.forward(x) x = x.reshape((-1,) + tuple(self.n_width) + (self.input_dim,) + tuple(self.n_height) + (1,) +(self.input_len,)).shape ", x.shape)

        x = x.transpose(2, 4)
        x = x.reshape((-1,) + (np.prod(self.n_height),) + (np.prod(self.n_width),) + (self.input_len,) + (self.input_dim,) )
        # (batch, [height], [width], lag, d_model)
        x = x.transpose(1,2)
        # (batch, [width], [height], lag, d_model)
        if self.verbose:
          print("-KUN-Encoder.forward(x)  x = x.transpose(1+len(self.n_width), 1+len(self.n_width)+len(self.n_height)+1).shape ", x.shape)

        x = x.reshape((-1, self.input_len, self.input_dim))
        # (batch * width * height, lag, d_model)
        if self.verbose:
          print("-KUN-Encoder.forward(x) x = x.reshape((-1, self.input_len, self.input_dim)).shape ", x.shape)


        x = self.layers(x)
        if self.verbose:
          print("-KUN-Encoder.forward(x) self.layers(x).shape ", x.shape)

        x = x.reshape((-1, self.output_len, self.output_dim))  # (batch * width * height , lag, d_model)
        if self.verbose:
          print("-KUN-Encoder.forward(x) x = x.reshape((-1, self.output_len, self.output_dim)).shape ", x.shape)

        # x = F.sigmoid(x)

        return x

class KUNetDecoder(nn.Module):
    def __init__(self, input_dim=128, input_len=4,
                 n_width=[1], n_height=[4, 4],
                 output_dim=128, output_len=1,
                 hidden_dim_list=[128]*3,  kernel_hidden_layer_list=[1]*3,
                 kernel_list=[Linear]*3, 
                 **kwargs):
        super(KUNetDecoder, self).__init__()
        self.hidden_dim_list = hidden_dim_list
        self.kernel_hidden_layer_list = kernel_hidden_layer_list
        self.n_width = n_width
        self.n_height = n_height
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.output_len = output_len

        self.unet_skip_concat = kwargs.get('unet_skip_concat', False)
        self.unet_skip = kwargs.get('unet_skip', None)

        self.verbose = kwargs.get('verbose', False)

        self.total_width = np.prod(self.n_width) * output_dim
        self.total_height = np.prod(self.n_height) * output_len


        assert isinstance(n_width, list)
        assert isinstance(n_height, list)
        assert isinstance(kernel_list, list)
        assert isinstance(hidden_dim_list, list)
        assert isinstance(kernel_hidden_layer_list, list)

        # Create model Optic Nerve transformer
        # lag_list = [lag, n_height_1, ...,n_height_n, n_width_1, n_width_n]
        #self.attention_layers_0 = MultiLayerModel(d_model, num_heads, num_layers, lag=lag, out_size=d_model)
        #self.attention_layers_1 = MultiLayerModel(d_model, num_heads, num_layers, lag=n_height, out_size=d_model)
        #self.attention_layers_2 = MultiLayerModel(d_model, num_heads, num_layers, lag=n_width, out_size=out_size)

        self.lag_list = [input_len] # [1]
        if not(len(self.n_width) == 1 and  self.n_width[0] == 1):
            self.lag_list =  self.lag_list + list(self.n_width)
        if not(len(self.n_height) == 1 and self.n_height[0] == 1):
            self.lag_list =  self.lag_list + list(self.n_height) #[10, 10]
        if (len(self.n_width) == 1 or  self.n_width[0] == 1) and \
           (len(self.n_height) == 1 or self.n_height[0] == 1) :
            self.lag_list =  self.lag_list + list(self.n_height) 

        # declear model
        kernel_list = list(reversed(kernel_list))
        kernel_hidden_layer_list = list(reversed(kernel_hidden_layer_list))
        hidden_dim_list = list(reversed(hidden_dim_list))
        kernel = kernel_list[0]
        kernel_hidden_layer = kernel_hidden_layer_list[0]
        hidden_dim = hidden_dim_list[0]
        self.layers = [KernelWrapper(kernel,
                        input_dim=input_dim, input_len=1,
                        output_dim=hidden_dim, output_len=self.lag_list[1],
                        **{"kernel_hidden_layer" : kernel_hidden_layer}, **kwargs)]

        multiple = 2 if self.unet_skip_concat else 1
        #print("self.concat", self.concat)
        self.layers = self.layers + [KernelWrapper(kernel_list[i],
                                        input_dim=hidden_dim_list[i-1]*multiple, input_len=1,
                                        output_dim=hidden_dim_list[i], output_len=l,
                                        **{"kernel_hidden_layer" : kernel_hidden_layer_list[i]}, **kwargs) \
                                           for i, l in enumerate(self.lag_list[2:], 1)]

        kernel = kernel_list[-1]
        kernel_hidden_layer = kernel_hidden_layer_list[-1]
        hidden_dim = hidden_dim_list[-1]
        self.layers.append(KernelWrapper(kernel,
                            input_dim=hidden_dim*multiple, input_len=1,
                            output_dim=output_dim, output_len=output_len,
                            **{"kernel_hidden_layer" : kernel_hidden_layer}, **kwargs)) # output_len = 10

        self.layers = nn.Sequential(*self.layers)

        for i, f in enumerate(self.layers):
          if i+1 < len(self.lag_list):
           f.next_layer_lag = self.lag_list[i+1]
           f.next_d_model = self.hidden_dim_list[i]
          else:
           f.next_d_model = self.output_dim

          f.transpose = True

    def forward(self, x):

        """
        # reshape
        1, shape : (batch, 1, d_model)
        2, shape : (batch*[width], 1, d_model)
        3, shape : (batch*[width]*[height], lag d_model)
        4, shape : (batch, [width], [height], lag, d_model)
        5, shape : (batch, [height]*lag, [width]*d_model)

        # layer 3 : process output
        3st step:  x -> model(x)
              (batch, lag=1, d_model)) -> (batch, out_size)

        # layer 2 : process width
        2st step:  x -> model(x)
                (batch, lag=1, d_model)) -> (batch, lag=width, d_model)
                output shape : (batch * width, lag=1, d_model))

        # layer 1 : process height
        1st step:  x -> model(x)
                (batch * [width] , lag=1, d_model) -> (batch * [width],  lag=[height], d_model)
                output shape : (batch * [width] * [height], lag=1, d_model)

        """
        x_shape = x.shape # (batch, 1, d_model)
        if self.verbose:
          print("-KUN-Decoder.forward(x) start x.shape ", x.shape)

        x = self.layers(x)
        if self.verbose:
          print("-KUN-Decoder.forward(x) self.layers(x).shape ", x.shape)  # (batch, lag, d_model)

        #x = x.transpose(1, 2) # (batch, d_model, lag)
        #print("x = x.transpose(1, 2).shape ", x.shape)

        #x = x.reshape((-1, self.output_len, self.output_dim))  # (batch * width * height , lag, d_model)
        #print("x.shape ", x.shape)

        # layer 1 : process height
        x = x.reshape((-1,) + (np.prod(self.n_width),) + (1, ) + (np.prod(self.n_height),) + (self.output_len,) + (self.output_dim,) ) # (batch, [width], d_model, [height], lag)
        if self.verbose:
          print("-KUN-Decoder.forward(x) x.reshape((-1,)+ tuple(self.n_width) + (1,) + tuple(self.n_height) + (self.output_dim,)  + (self.output_len,)).reshape ", x.shape)

        x = x.transpose(1, 3)
        # (batch, [width], [height], lag, d_model)
        if self.verbose:
          print("-KUN-Decoder.forward(x) x.transpose(2+len(self.n_width), 2+len(self.n_width)+2+len(self.n_height)).shape ", x.shape)

        x = x.transpose(2, 4)
        # (batch, [height]*lag, [width]*d_model)

        x = x.reshape((x_shape[0], self.total_height, self.total_width))
        # (batch, height * lag,   width * d_model)
        if self.verbose:
          print("-KUN-Decoder.forward(x) x.reshape((x_shape[0], self.total_width, self.total_height*self.input_len)).shape ", x.shape)


        return x
 

class KUNetEncoderDecoder(nn.Module):
    def __init__(self, input_dim=128, input_len=4,
                 n_width=[1], n_height=[4, 4],
                 latent_dim=128, latent_len=1,
                 output_dim=128, output_len=4,
                 hidden_dim_list=[128]*3,
                 kernel_list=[Linear]*3, kernel_hidden_layer_list=[1, 1, 1],
                 **kwargs):
        super(KUNetEncoderDecoder, self).__init__()

        self.unet_skip = kwargs.get('unet_skip', True)
        self.verbose = kwargs.get('verbose', False)


        self.encoder = KUNetEncoder(input_dim=input_dim, input_len=input_len,
                            n_width=n_width, n_height=n_height,
                            output_dim=latent_dim, output_len=latent_len,
                            hidden_dim_list=hidden_dim_list, kernel_hidden_layer_list=kernel_hidden_layer_list,
                            kernel_list=kernel_list, **kwargs)

        self.decoder = KUNetDecoder(input_dim=latent_dim, input_len=latent_len,
                            n_width=n_width, n_height=n_height,
                            output_dim=output_dim, output_len=output_len,
                            hidden_dim_list=hidden_dim_list, kernel_hidden_layer_list=kernel_hidden_layer_list,
                            kernel_list=kernel_list, **kwargs)

        self.latent_condition = None #torch.zeros((1,1,128))
        self.encoder_condition = None #torch.zeros((1,1,128))
        self.latent = None #torch.zeros((B, 1,128))

    def forward(self, x):
        #seq_last = x[:,-1:,:]
        #print("seq_last.shape ", seq_last.shape)
        #x = x - seq_last

        x_shape = x.shape # (batch, lag, d_model)
        #print("start x.shape ", x.shape)

        #x = x.transpose(1, 2) # (batch, d_model, lag)
        #print(" x.transpose(1, 2).shape ", x.shape)

        encoder = self.encoder
        decoder = self.decoder

        if self.encoder_condition is not None:
          x[:, 0:1, :] = self.encoder_condition

        z = encoder(x)
        self.latent = z
        if self.latent_condition is not None:
          z[:, :, 128:256] = self.latent_condition

        if self.unet_skip:
          #print("self.encoder(x).shape ", z.shape)
          _unet_skip_output_list = []
          for f in encoder.layers:
            #print(f.transpose)
            #print("f._unet_skip_output.shape", f._unet_skip_output.shape)
            _unet_skip_output_list.append(f._unet_skip_output)

          for i, f in enumerate(decoder.layers):
            #print("i=", i)
            if i > 0 and i < len(_unet_skip_output_list):
              #print("i >=1 and i < len(_unet_skip_output_list)", i >1 and i < len(_unet_skip_output_list))
              #print("i ", i,", -1-i ", -1-i)
              f._unet_skip_input = _unet_skip_output_list[-1-i]
              #print("f._unet_skip_input.shape", f._unet_skip_input.shape)
        y = decoder(z)
          #print("self.decoder(z).shape ", y.shape)
        #y = y.transpose(1, 2) # (batch, lag, d_model)
        #print(" y.transpose(1, 2).shape ", y.shape)
        #y = y + seq_+last
        #y = F.relu(yx)
        #y = F.relu(y)
        return y

class KUNet(nn.Module):
    def __init__(self, input_dim=1, input_len=8,
                 n_width=[1], n_height=[1],
                 latent_dim=128, latent_len=1,
                 output_dim=1, output_len=8,
                 hidden_dim_list=[128], kernel_hidden_layer_list=[1],
                 kernel_list=[Linear],
                 **kwargs):

        super(KUNet, self).__init__()

        self.input_dim, self.input_len, self.n_width, self.n_height, \
                 self.latent_dim, self.latent_len, self.output_dim, self.output_len, \
                 self.hidden_dim_list, self.kernel_hidden_layer_list, self.kernel_list =  \
                      input_dim, input_len, n_width, n_height, \
                      latent_dim, latent_len, output_dim, output_len, \
                      hidden_dim_list, kernel_hidden_layer_list, kernel_list \

        self.inverse_norm = kwargs.get('inverse_norm', False)
        self.mean_norm = kwargs.get('mean_norm', False)
        self.chanel_independent = kwargs.get('chanel_independent', False)
        self.residual = kwargs.get('residual', False)
        self.verbose = kwargs.get('verbose', False)


        if isinstance(n_width, int):
           n_width = [n_width]
        if isinstance(n_height, int):
           n_height = [n_height]

        n_enc_layers = 1 + len(n_width) + len(n_height) - (1 if np.prod(n_width) == 1 else 0) - (1 if np.prod(n_height) == 1 else 0)
        if not isinstance(hidden_dim_list, list):
           hidden_dim_list = [hidden_dim_list] * n_enc_layers
        if not isinstance(kernel_hidden_layer_list, list):
           kernel_hidden_layer_list = [int(i) for i in kernel_hidden_layer_list]
        if not isinstance(kernel_list, list):
           kernel_list = [Linear if i == 0 else kernel_list for i in kernel_hidden_layer_list]

        self.model = KUNetEncoderDecoder(input_dim=input_dim, input_len=input_len,
                                        n_width=n_width, n_height=n_height,
                                        latent_dim=latent_dim, latent_len=latent_len,
                                        output_dim=output_dim, output_len=output_len,
                                        hidden_dim_list=hidden_dim_list,
                                        kernel_list=kernel_list, kernel_hidden_layer_list=kernel_hidden_layer_list,
                                        **kwargs)

        self.output = None #torch.zeros((B, 1,128))

    def forward(self, x):
        B, L, M = x.shape
        if self.residual:
          res = x
        # Instance normalization - pahse 1
        if self.inverse_norm:
          seq_std, seq_mean = torch.std_mean(x, dim=1, keepdim=True)
          if self.mean_norm : # noly use mean normalisation
            seq_std = 1
          x = (x - seq_mean) / (seq_std + 0.000001)

        # Chanel Independent - pahse 1
        if self.chanel_independent :
            x = x.transpose(2, 1)
            x = x.reshape(B * M, L, 1)  # Reshape the input to (batch_size, input_len)

        output = self.model(x)

        # Chanel Independent - pahse 2
        if self.chanel_independent :
            output = output.reshape(B, M, L)  # Reshape the input to (batch_size, input_len)
            output = output.transpose(2, 1)

        # Instance normalization - pahse 2
        if self.inverse_norm:
          output = output * (seq_std + 0.000001)   + seq_mean

        if self.residual:
          output = F.relu( output + res)

        self.output = output
        return output

    def set_latent_conditions(self, latent):
        self.model.latent_condition = latent


class Projector(nn.Module):
  def __init__(self, input_shape, output_shape, dim) -> None:
     super().__init__()
     self.input_shape = input_shape
     self.output_shape = output_shape
     self.dim = dim
     self.model = nn.Linear(input_shape[dim], output_shape[dim])

  def forward(self, x):
    if len(self.input_shape) != (self.dim + 1):
      x = torch.transpose(x, self.dim, -1)
    else:
      x = self.model(x)
    #if len(self.input_shape) != (self.dim + 1):
     # x = torch.transpose(x, self.dim, -1)
    return  x
