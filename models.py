import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from kernels import LSTM, Transformer

class KernelWrapper(nn.Module):
    def __init__(self, kernal, input_dim, input_len, output_dim=1, output_len=1, num_hidden_layers=1, use_relu=True, drop_out_p=0.01):
        super(KernelWrapper, self).__init__()

        # kernal : kernal(input_dim, input_len, output_dim, output_len)

        self.input_dim, self.input_len, self.output_dim, self.output_len = \
                        input_dim, input_len, output_dim, output_len

        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_list = []
        self.layers = []
        in_size = input_len*input_dim
        out_size = output_len*output_dim

        self.is_linear_kernel = False

        if in_size >= out_size:
          gap = int((in_size - out_size) / (num_hidden_layers + 1))
          self.hidden_size_list = [in_size - i * gap for i in range(1, num_hidden_layers + 1)]
        else:
          gap = int((out_size - in_size) / (num_hidden_layers + 1))
          self.hidden_size_list = [in_size + i * gap for i in range(1, num_hidden_layers + 1)]

        if "Linear" in kernal:

          for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(in_size, self.hidden_size_list[i]))
            if use_relu:
              self.layers.append(nn.Tanh())
              #self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(drop_out_p))

            in_size = self.hidden_size_list[i]

          self.layers.append(nn.Linear(in_size, out_size))
          self.is_linear_kernel = True
        elif "Transformer" in kernal:
          self.layers.append(Transformer(input_dim, input_len, output_dim, output_len, num_hidden_layers))

        elif "LSTM" in kernal:
          self.layers.append(LSTM(input_dim, input_len, output_dim, output_len, num_hidden_layers))

        else:
          print("kernal", kernal, "is not recognized")

        self.layers = nn.Sequential(* self.layers)

        self.next_layer_lag = 0
        self.next_d_model = 0
        self.shuffuled_index = []

        self._unet_skip_output = None
        self._unet_skip_input = None

        #self.linear_unet_skip_input = nn.Linear(self.input_dim, self.input_dim)

        self.transpose=False

    def f(self, x):
        x = self.layers(x)
        return x

    def forward(self, x, train = False):

        # Add U-Net encoder hidden vector in input of decoder
        if self.transpose and self._unet_skip_input is not None:
          if np.prod(x.shape) == np.prod(self._unet_skip_input.shape):
            x = x + self._unet_skip_input.reshape(x.shape)
            
        # Assert and Reshape the input 
        x = x.reshape(-1, self.input_len, self.input_dim)   
        if self.is_linear_kernel:
          x = x.reshape(-1, self.input_len * self.input_dim)
           

        # Compute with kernel f
        x = self.f(x)

        # Assert the output shape 
        if self.is_linear_kernel:
          x = x.reshape(-1, self.output_len, self.output_dim)
        assert x.shape[1] == self.output_len and x.shape[2] == self.output_dim

        # Save hidden vector in U-Net encoder 
        if not self.transpose:
          self._unet_skip_output = x
        
        return x

class KUNetEncoder(nn.Module):
    def __init__(self, input_dim, input_len, n_width=1, n_height=1, output_dim=1, output_len=1, hidden_dim=20, num_hidden_layers=1, kernal_model=nn.Linear):
        super(KUNetEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_width = n_width
        self.n_height = n_height
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.output_len = output_len

        if isinstance(n_width, int):
          self.n_width = [n_width]
        if isinstance(n_height, int):
          self.n_height = [n_height]

        # Create layers in KUN encoder
        # lag_list = [lag, n_height_1, ...,n_height_n, n_width_1, n_width_n]
        
        self.lag_list = [input_len]
        if not(len(self.n_height) == 1 and self.n_height[0] ==1):
            self.lag_list =  self.lag_list + list(reversed(self.n_height))
        if not(len(self.n_width) == 1 and  self.n_width[0] ==1):
            self.lag_list =  self.lag_list + list(reversed(self.n_width))
        print(self.lag_list)
        print(hidden_dim)

        if isinstance(kernal_model, list):
          kernal_model_list = kernal_model
          num_hidden_layers_list = num_hidden_layers
          hidden_dim_list = hidden_dim
          num_hidden_layers = num_hidden_layers_list[0]
          kernal_model = kernal_model_list[0]
          hidden_dim = hidden_dim_list[0]
          self.layers = [KernelWrapper(kernal_model, input_dim=input_dim, input_len=self.lag_list[0], output_dim=hidden_dim, output_len=1, num_hidden_layers=num_hidden_layers)]
          self.layers = self.layers + [KernelWrapper(kernal_model_list[i+1], input_dim=hidden_dim_list[i], input_len=l, output_dim=hidden_dim_list[i+1], output_len=1, num_hidden_layers=num_hidden_layers_list[i+1]) for i, l in enumerate(self.lag_list[1:-1])]

          kernal_model = kernal_model_list[len(self.layers)]
          num_hidden_layers = num_hidden_layers_list[len(self.layers)]
          hidden_dim = hidden_dim_list[len(self.layers)-1]
          self.layers.append(KernelWrapper(kernal_model, input_dim=hidden_dim, input_len=self.lag_list[-1], output_dim=output_dim, output_len=output_len, num_hidden_layers=num_hidden_layers))
        else:
          self.layers = [KernelWrapper(kernal_model,input_dim=input_dim, input_len=self.lag_list[0], output_dim=hidden_dim, output_len=1, num_hidden_layers=num_hidden_layers)]
          self.layers = self.layers + [KernelWrapper(kernal_model,input_dim=hidden_dim, input_len=l, output_dim=hidden_dim, output_len=1, num_hidden_layers=num_hidden_layers) for l in self.lag_list[1:-1]]
          self.layers.append(KernelWrapper(kernal_model,input_dim=hidden_dim, input_len=self.lag_list[-1], output_dim=output_dim, output_len=output_len, num_hidden_layers=num_hidden_layers))

        self.layers = nn.Sequential(* self.layers)

        for i, f in enumerate(self.layers):
          if i+1 < len(self.lag_list):
           f.next_layer_lag = self.lag_list[i+1]
           f.next_d_model = self.hidden_dim[i]
          else:
           f.next_d_model = self.output_dim


    def forward(self, x):
        """ 
        # Steps :
        1, input shape of x : (B, L, M) = (batch, [M_height]*lag, [M_width]*d_model)
        2, reshape : (batch, [M_height], lag, [M_width], 1, d_model)

        3, transpose : (batch, [M_height], 1, [M_width], lag, d_model)
        4, transpose : (batch, [M_width], 1, [M_height], lag, d_model)
        5, reshape : (batch*[M_width]*[M_height], lag, d_model)

        6, compute : x = f(x)
        7, return shape of x : (batch, 1, d_model)

        """ 
        # Step 2:
        # (batch, [height]*lag, [width]*d_model)
        # -> (batch, [M_height], lag, [M_width], 1, d_model)
        x = x.reshape((-1,)+ (np.prod(self.n_height),) +(self.input_len,) + (np.prod(self.n_width),) + (1,) + (self.input_dim,))

        # Step 3:
        # (batch, [M_height], lag, [M_width], 1, d_model)
        #  -> (batch, [M_height], 1, [M_width], lag, d_model)
        x = x.transpose(2, 4)

        # Step 4:
        # (batch, [M_height], 1, [M_width], lag, d_model)
        #  -> (batch, [M_width], 1, [M_height], lag, d_model)
        x = x.transpose(1,3)

        # Step 5:
        # (batch, [M_width], 1, [M_height], lag, d_model)
        #  -> (batch*[M_width]*[M_height], lag, d_model)
        x = x.reshape((-1, self.input_len, self.input_dim))

        # Step 6:
        # (batch*[M_width]*[M_height], lag, d_model)
        #  -> (batch, 1, d_model)
        x = self.layers(x)

        #x = x.reshape((-1, self.output_len, self.output_dim))  # (batch * width * height , lag, d_model)

        return x
    
class KUNetDecoder(nn.Module):
    def __init__(self, input_dim, input_len=1, n_width=1, n_height=1, output_dim=1, output_len=1, hidden_dim=20, num_hidden_layers=1, kernal_model=nn.Linear):
        super(KUNetDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_width = n_width
        self.n_height = n_height
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.output_len = output_len


        if isinstance(n_width, int):
          self.n_width = [n_width]
        if isinstance(n_height, int):
          self.n_height = [n_height]

        self.total_width = np.prod(self.n_width) * output_dim
        self.total_height = np.prod(self.n_height) * output_len


        # Create layers in KUN decoder
        # lag_list = [1,  n_width_1,..., n_width_n, n_height_1, ...,n_height_n]

        self.lag_list = [input_len] # [1]
        if not(len(self.n_width) == 1 and  self.n_width[0] ==1):
            self.lag_list =  self.lag_list + list(self.n_width)
        if not(len(self.n_height) == 1 and self.n_height[0] ==1):
            self.lag_list =  self.lag_list + list(self.n_height) #[10, 10]

        if isinstance(kernal_model, list):
          kernal_model_list = list(reversed(kernal_model))
          num_hidden_layers_list = list(reversed(num_hidden_layers))
          hidden_dim_list = list(reversed(hidden_dim))
          num_hidden_layers = num_hidden_layers_list[0]
          kernal_model = kernal_model_list[0]
          hidden_dim = hidden_dim_list[0]
          self.layers = [KernelWrapper(kernal_model, input_dim=input_dim, input_len=1, output_dim=hidden_dim, output_len=self.lag_list[1],num_hidden_layers =num_hidden_layers)]

          self.layers = self.layers + [KernelWrapper(kernal_model_list[i+1], input_dim=hidden_dim_list[i], input_len=1, output_dim=hidden_dim_list[i+1], output_len=l,num_hidden_layers=num_hidden_layers_list[i+1]) for i, l in enumerate(self.lag_list[2:])]

          kernal_model = kernal_model_list[-1]
          num_hidden_layers = num_hidden_layers_list[-1]
          hidden_dim = hidden_dim_list[-1]
          self.layers.append(KernelWrapper(kernal_model, input_dim=hidden_dim, input_len=1, output_dim=output_dim, output_len=output_len,num_hidden_layers=num_hidden_layers)) # output_len = 10
        else:
          self.layers = [KernelWrapper(kernal_model, input_dim=input_dim, input_len=1, output_dim=hidden_dim, output_len=self.lag_list[1],num_hidden_layers=num_hidden_layers)]
          self.layers = self.layers + [KernelWrapper(kernal_model, input_dim=hidden_dim, input_len=1, output_dim=hidden_dim, output_len=l,num_hidden_layers=num_hidden_layers) for l in self.lag_list[2:]]
          self.layers.append(KernelWrapper(kernal_model, input_dim=hidden_dim, input_len=1, output_dim=output_dim, output_len=output_len,num_hidden_layers=num_hidden_layers)) # output_len = 10

        self.layers = nn.Sequential(*self.layers)

        for i, f in enumerate(self.layers):
          if i+1 < len(self.lag_list):
           f.next_layer_lag = self.lag_list[i+1]
           f.next_d_model = self.hidden_dim[i]
          else:
           f.next_d_model = self.output_dim

          f.transpose = True



    def forward(self, x):
        """ 
        # Steps :
        1, input shape of x : (batch, 1, d_model)
        2, compute : x = f(x) : (batch*[M_width]*[M_height], lag, d_model)

        3, reshape : (batch, [M_width], 1, [M_height], lag, d_model)
        4, transpose : (batch, [M_height], 1, [M_width], lag, d_model)
        5, transpose : (batch, [M_height], lag, [M_width], 1, d_model)
        6, reshape : (batch, [M_height]*lag, [M_width]*d_model)

        7, return shape of x : (B, L, M)
        """
        # Step 2:
        x = self.layers(x)

        # Step 3:
        # (batch*[M_width]*[M_height], lag, d_model)
        # -> (batch, [M_width], 1, [M_height], lag, d_model)
        x = x.reshape((-1,) + (np.prod(self.n_width),) + (1, ) + (np.prod(self.n_height),) + (self.output_len,) + (self.output_dim,) ) # (batch, [width], d_model, [height], lag)

        # Step 4:
        # (batch, [M_width], 1, [M_height], lag, d_model)
        #  -> (batch, [M_height], 1, [M_width], lag, d_model)
        x = x.transpose(1, 3)

        # Step 5:
        # (batch, [M_height], 1, [M_width], lag, d_model)
        #  -> (batch, [M_height], lag, [M_width], 1, d_model)
        x = x.transpose(2, 4)

        # Step 6:
        # (batch, [M_height], lag, [M_width], 1, d_model)
        # -> (batch, [height]*lag, [width]*d_model)
        x = x.reshape((-1, self.total_height, self.total_width))

        return x

class KUNetEncoderDecoder(nn.Module):
    def __init__(self, input_dim=7, input_len=3, n_width=1, n_height=[10, 6, 4], output_dim=10, output_len=1, hidden_dim=10, num_hidden_layers=0, kernal_model=nn.Linear):
        super(KUNetEncoderDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_width = n_width
        if isinstance(n_height[0], list):
          n_height_in, n_height_out = n_height[0], n_height[1]
        else:
          n_height_in, n_height_out = n_height, n_height
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.output_len = output_len

        self.use_unet_skip=True

        self.encoder = KUNetEncoder(input_dim=input_dim, input_len=input_len, n_width=n_width, n_height=n_height_in, output_dim=output_dim, output_len=output_len, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, kernal_model=kernal_model)
        self.decoder = KUNetDecoder(input_dim=output_dim, input_len=output_len, n_width=n_width, n_height=n_height_out, output_dim=input_dim, output_len=input_len, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, kernal_model=kernal_model)

    def forward(self, x):
        # Chanel Independent setting by default
        # x.shape : CI: (B*M, L, 1), no CI: (B, L, M)

        encoder = self.encoder
        decoder = self.decoder
        
        # Forward pass with KUN encoder
        z = encoder(x)

        # Use U-Net Skip to update the hidden vector at each level.
        if self.use_unet_skip:
          _unet_skip_output_list = []

          # Retrive [M_{h, 1}, ..., M_{h, n}] for n layers from encoder
          for f in encoder.layers: 
            _unet_skip_output_list.append(f._unet_skip_output) 

          # Update [M_{h, 1}, ..., M_{h, n}] for n layers for decoder
          for i, f in enumerate(decoder.layers):  # first layer is near the latent vector so skip it.
            if i > 0: # and i < len(_unet_skip_output_list):
              f._unet_skip_input = _unet_skip_output_list[-1-i]
              
        # Forward pass with KUN decoder
        y = decoder(z)
        return y

class NKUNet(nn.Module):
    """
    Normalized Kernel U-Net
    """
    def __init__(self, args=None, input_dim=1, input_len=3, n_width=[7], n_height=[5, 4, 4, 3], output_dim=10, output_len=1, hidden_dim=10, num_hidden_layers=0, kernal_model=None):
        super(NKUNet, self).__init__()
        if args is not None:
            num_kun = 1 if args.num_kun is None else args.num_kun
            num_hidden_layers = num_hidden_layers if args.num_hidden_layers is None else args.num_hidden_layers
            kernal_model = kernal_model if args.kernal_model is None else args.kernal_model
            input_dim = input_dim if args.input_dim is None else args.input_dim
            input_len = input_len if args.input_len is None else args.input_len
            n_width = n_width if args.n_width is None else args.n_width
            n_height = n_height if args.n_height is None else args.n_height
            output_dim = output_dim if args.output_dim is None else args.output_dim
            hidden_dim = hidden_dim if args.hidden_dim is None else args.hidden_dim
            use_chanel_independence = args.use_chanel_independence
            use_instance_norm = args.use_instance_norm
        self.use_chanel_independence = use_chanel_independence
        self.use_instance_norm = use_instance_norm

        self.linear_list = nn.ModuleList([KUNetEncoderDecoder(input_dim, input_len, n_width, n_height, output_dim, output_len, hidden_dim, num_hidden_layers, kernal_model) for _ in range(num_kun)])

    def forward(self, x): 
        # Input x : [Batch, Length, Channel(M)]

        # Instance normalization, else use Mean Normalization
        if self.use_instance_norm:
          seq_std, seq_mean = torch.std_mean(x, dim=1, keepdim=True)
          x = (x - seq_mean) / (seq_std + 0.000001)
        else:
          seq_mean = x.mean(1, True).detach()
          x = x - seq_mean
        x_shape = x.shape

        # Chanel Independent
        # CI: (B*M, L, 1), no CI: (B, L, M)
        if self.use_chanel_independence:
          x = x.transpose(1, 2) # (B, M, L)
          x = x.reshape(-1, 1, x_shape[1]) # (B*M, 1, L)
          x = x.transpose(1,2) # (B*M, L, 1)
          
        # Forward pass KUN
        for i, f in enumerate(self.linear_list):
          if i==0:
            y = f(x)
          else:
            y = y+f(y) 
        
        # Reconstruct Chanel Independent
        if self.use_chanel_independence: # (B*M, L, 1)
          y = y.reshape(x_shape[0], x_shape[2], -1) # (B, M, L)
          y = y.transpose(1,2) # (B, L, M)
        
        # Reverse Instance Normalization, else use Reverse Mean Normalization
        if self.use_instance_norm:
          y = y * (seq_std + 0.000001)   + seq_mean
        else:
          y = y + seq_mean
        return y # [Batch, Length, Channel]