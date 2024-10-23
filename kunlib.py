
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class KernelWrapper(nn.Module):
    def __init__(self, kernal, input_dim, input_len, 
                 output_dim=1, output_len=1, 
                 num_hidden_layers=1, use_relu=True, drop_out_p=0.01, 
                 mode="concate", verbose=False):
        super(KernelWrapper, self).__init__()

        # kernal : kernal(input_dim, input_len, output_dim, output_len)

        self.input_dim, self.input_len, self.output_dim, self.output_len = \
                        input_dim, input_len, output_dim, output_len
        self.verbose = verbose
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_list = []
        self.layers = []
        in_size = input_len * input_dim
        out_size = output_len * output_dim

        self.is_linear_kernel = False

        if in_size >= out_size:
          gap = int((in_size - out_size) / (num_hidden_layers + 1))
          self.hidden_size_list = [in_size - i * gap for i in range(1, num_hidden_layers + 1)]
        else:
          gap = int((out_size - in_size) / (num_hidden_layers + 1))
          self.hidden_size_list = [in_size + i * gap for i in range(1, num_hidden_layers + 1)]

        if "Linear" in kernal.__name__:

          for i in range(num_hidden_layers):
            self.layers.append(kernal(in_size, self.hidden_size_list[i]))
            if use_relu:
              self.layers.append(nn.Tanh())
              #self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(drop_out_p))

            in_size = self.hidden_size_list[i]

          self.layers.append(kernal(in_size, out_size))
          self.is_linear_kernel = True
        elif "Transformer" in kernal.__name__:
          self.layers.append(kernal(input_dim, input_len, output_dim, output_len, num_hidden_layers))

        elif "LSTM" in kernal.__name__:
          self.layers.append(kernal(input_dim, input_len, output_dim, output_len, num_hidden_layers))

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
        self.concat = True if mode == "concate" else False

    def f(self, x):
        if self.verbose : 
          print("---KernelWrapper.f(x) Input x.shape: ", x.shape)
        x = self.layers(x)
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
          #print("--x.shape", x.shape)
          if np.prod(x.shape) == np.prod(self._unet_skip_input.shape):
            #print( self.concat, x.shape)
            if self.concat:
              x = torch.cat([x, self._unet_skip_input.reshape(x.shape)], dim=-1)
              #print( "after, ", self.concat, x.shape)
            else:
              x = x + self._unet_skip_input.reshape(x.shape)
            #print("_unet_skip_input.shape", self._unet_skip_input.shape)
          #x[len(self._unet_skip_input):] = x[len(self._unet_skip_input):] + self._unet_skip_input
        #x = x.transpose(1, 2)   # # (batch, d_model , lag) to (batch, lag, d_model)

        #print("x.shape", x.shape)
        x = x.reshape(-1, self.input_len, self.input_dim)
        if self.is_linear_kernel:
          x = x.reshape(-1, self.input_len * self.input_dim)

        #print("x.reshape", x.shape)
        x = self.f(x)

        if self.is_linear_kernel:
          x = x.reshape(-1, self.output_len, self.output_dim)

        #print("x.shape", x.shape)
        assert x.shape[1] == self.output_len and x.shape[2] == self.output_dim

        if not self.transpose:
          self._unet_skip_output = x
        return x


class KUNetEncoder(nn.Module):
    def __init__(self, input_dim, input_len, 
                 n_width=1, n_height=1, 
                 output_dim=1, output_len=1, 
                 hidden_dim=20, num_hidden_layers=1, 
                 kernal_model=nn.Linear, verbose=False):
        super(KUNetEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_width = n_width
        self.n_height = n_height
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.output_len = output_len

        self.verbose=verbose

        if isinstance(n_width, int):
          self.n_width = [n_width]
        if isinstance(n_height, int):
          self.n_height = [n_height]

        # Create model Optic Nerve transformer
        # lag_list = [lag, n_height_1, ...,n_height_n, n_width_1, n_width_n]
        #self.attention_layers_0 = MultiLayerModel(d_model, num_heads, num_layers, lag=lag, out_size=d_model)
        #self.attention_layers_1 = MultiLayerModel(d_model, num_heads, num_layers, lag=n_height, out_size=d_model)
        #self.attention_layers_2 = MultiLayerModel(d_model, num_heads, num_layers, lag=n_width, out_size=out_size)

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
          self.layers = [KernelWrapper(kernal_model, 
                                       input_dim=input_dim, input_len=self.lag_list[0], 
                                       output_dim=hidden_dim, output_len=1, 
                                       num_hidden_layers=num_hidden_layers, verbose=verbose)]
          self.layers = self.layers + [KernelWrapper(kernal_model_list[i+1], 
                                                     input_dim=hidden_dim_list[i], input_len=l, 
                                                     output_dim=hidden_dim_list[i+1], output_len=1, 
                                                     num_hidden_layers=num_hidden_layers_list[i+1], verbose=verbose) for i, l in enumerate(self.lag_list[1:-1])]

          kernal_model = kernal_model_list[len(self.layers)]
          num_hidden_layers = num_hidden_layers_list[len(self.layers)]
          hidden_dim = hidden_dim_list[len(self.layers)-1]
          self.layers.append(KernelWrapper(kernal_model, 
                                input_dim=hidden_dim, input_len=self.lag_list[-1], 
                                output_dim=output_dim, output_len=output_len, 
                                num_hidden_layers=num_hidden_layers, verbose=verbose))
        else:
          self.layers = [KernelWrapper(kernal_model, 
                            input_dim=input_dim, input_len=self.lag_list[0], 
                            output_dim=hidden_dim, output_len=1, 
                            num_hidden_layers=num_hidden_layers, verbose=verbose)]
          self.layers = self.layers + [KernelWrapper(kernal_model,
                                          input_dim=hidden_dim, input_len=l, 
                                          output_dim=hidden_dim, output_len=1, 
                                          num_hidden_layers=num_hidden_layers, verbose=verbose) for l in self.lag_list[1:-1]]
          self.layers.append(KernelWrapper(kernal_model, 
                                input_dim=hidden_dim, input_len=self.lag_list[-1], 
                                output_dim=output_dim, output_len=output_len, 
                                num_hidden_layers=num_hidden_layers, verbose=verbose))

        self.layers = nn.Sequential(* self.layers)

        for i, f in enumerate(self.layers):
          if i+1 < len(self.lag_list):
           f.next_layer_lag = self.lag_list[i+1]
           f.next_d_model = self.hidden_dim[i]
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
    def __init__(self, input_dim, input_len=1, 
                 n_width=1, n_height=1, 
                 output_dim=1, output_len=1, 
                 hidden_dim=20, num_hidden_layers=1, 
                 kernal_model=nn.Linear, skip_conn=False, 
                 concat=False, verbose=False):
        super(KUNetDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.n_width = n_width
        self.n_height = n_height
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.output_len = output_len

        self.concat = concat

        self.verbose=verbose

        if isinstance(n_width, int):
          self.n_width = [n_width]
        if isinstance(n_height, int):
          self.n_height = [n_height]

        self.total_width = np.prod(self.n_width) * output_dim
        self.total_height = np.prod(self.n_height) * output_len


        # Create model Optic Nerve transformer
        # lag_list = [lag, n_height_1, ...,n_height_n, n_width_1, n_width_n]
        #self.attention_layers_0 = MultiLayerModel(d_model, num_heads, num_layers, lag=lag, out_size=d_model)
        #self.attention_layers_1 = MultiLayerModel(d_model, num_heads, num_layers, lag=n_height, out_size=d_model)
        #self.attention_layers_2 = MultiLayerModel(d_model, num_heads, num_layers, lag=n_width, out_size=out_size)

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
          self.layers = [KernelWrapper(kernal_model, 
                            input_dim=input_dim, input_len=1, 
                            output_dim=hidden_dim, output_len=self.lag_list[1], 
                            num_hidden_layers =num_hidden_layers, verbose=verbose)]

          multiple = 2 if self.concat else 1
          #print("self.concat", self.concat)
          self.layers = self.layers + [KernelWrapper(kernal_model_list[i+1], 
                                            input_dim=hidden_dim_list[i]*multiple, input_len=1, 
                                            output_dim=hidden_dim_list[i+1], output_len=l, 
                                            num_hidden_layers=num_hidden_layers_list[i+1], verbose=verbose) for i, l in enumerate(self.lag_list[2:])]

          kernal_model = kernal_model_list[-1]
          num_hidden_layers = num_hidden_layers_list[-1]
          hidden_dim = hidden_dim_list[-1]
          self.layers.append(KernelWrapper(kernal_model, 
                                input_dim=hidden_dim*multiple, input_len=1, 
                                output_dim=output_dim, output_len=output_len, 
                                num_hidden_layers=num_hidden_layers, verbose=verbose)) # output_len = 10
        else:
          self.layers = [KernelWrapper(kernal_model, 
                              input_dim=input_dim, input_len=1, 
                              output_dim=hidden_dim, output_len=self.lag_list[1], 
                              num_hidden_layers=num_hidden_layers)]
          self.layers = self.layers + [KernelWrapper(kernal_model, 
                                            input_dim=hidden_dim, input_len=1, 
                                            output_dim=hidden_dim, output_len=l, 
                                            num_hidden_layers=num_hidden_layers, verbose=verbose) for l in self.lag_list[2:]]
          self.layers.append(KernelWrapper(kernal_model, 
                                    input_dim=hidden_dim, input_len=1, 
                                    output_dim=output_dim, output_len=output_len,
                                    num_hidden_layers=num_hidden_layers, verbose=verbose)) # output_len = 10

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
    def __init__(self, input_dim=7, input_len=3, n_width=1, n_height=[10, 6, 4], 
                 latent_dim=10, latent_len=1, output_dim=1, output_len=1, 
                 hidden_dim=10, num_hidden_layers=0, 
                 kernal_model=nn.Linear, non_linear_kernel_pos="011",
                 skip_conn=True, skip_mode='concat', verbose=False):
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

        self.use_unet_skip = skip_conn
        self.skip_mode = skip_mode
        self.concat = True if skip_mode=="concat" and skip_conn else False

        self.encoder = KUNetEncoder(input_dim=input_dim, input_len=input_len, 
                            n_width=n_width, n_height=n_height_in, 
                            output_dim=latent_dim, output_len=latent_len, 
                            hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, 
                            kernal_model=kernal_model, verbose=verbose)
        self.decoder = KUNetDecoder(input_dim=latent_dim, input_len=latent_len, 
                            n_width=n_width, n_height=n_height_out, 
                            output_dim=output_dim, output_len=output_len, 
                            hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, 
                            kernal_model=kernal_model, skip_conn=skip_conn,
                            concat=self.concat, verbose=verbose)
        self.latent_condition = None #torch.zeros((1,1,128))

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
        z = encoder(x)
        if self.latent_condition is not None:
          z[:, :, 128:256] = self.latent_condition

        if self.use_unet_skip:
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
        #y = y + seq_last
        #y = F.relu(y+x)
        #y = F.relu(y)
        return y

class NKUNet(nn.Module):
    """
    Normalized Kernel U-Net
    """
    def __init__(self, args, input_dim=1, input_len=3, n_width=[7], n_height=[5, 4, 4, 3], latent_dim=1, latent_len=10, output_dim=10, output_len=1, hidden_dim=10, num_hidden_layers=0, kernal_model=None):
        super(NKUNet, self).__init__()
        num_layers = 2 if args.num_layers is None else args.num_layers
        num_hidden_layers = num_hidden_layers if args.num_hidden_layers is None else args.num_hidden_layers
        kernal_model = kernal_model if args.kernal_model is None else args.kernal_model
        input_dim = input_dim if args.input_dim is None else args.input_dim
        input_len = input_len if args.input_len is None else args.input_len
        n_width = n_width if args.n_width is None else args.n_width
        n_height = n_height if args.n_height is None else args.n_height
        output_dim = output_dim if args.output_dim is None else args.output_dim
        output_len = output_len if args.output_len is None else args.output_len
        latent_dim = latent_dim if args.latent_dim is None else args.latent_dim
        latent_len = latent_len if args.latent_len is None else args.latent_len
        hidden_dim = hidden_dim if args.hidden_dim is None else args.hidden_dim
        self.use_chanel_independence = args.use_chanel_independence
        self.use_instance_norm = args.use_instance_norm

        self.args = args
        self.T = self.args.pred_len

        self.linear_list = nn.ModuleList([KUNetEncoderDecoder(input_dim, input_len, n_width, n_height, latent_dim, latent_len, output_dim, output_len, hidden_dim, num_hidden_layers, kernal_model) for _ in range(num_layers)])

    def forward(self, src):
        B, L, M = src.shape
        src = src.transpose(2, 1)
        output = src.reshape(B * M, L)  # Reshape the input to (batch_size, input_len)

        for i, f in enumerate(self.linear_list):
            output = f(output)
        output = output.reshape(B, M, L)  # Reshape the input to (batch_size, input_len)
        output = output.transpose(2, 1)
        return output # [Batch, Output length, Channel]

class KUNet(nn.Module):
    def __init__(self, input_dim=1, input_len=8, 
                 n_width=[1], n_height=[8,8], 
                 latent_dim=128, latent_len=1, 
                 output_dim=1, output_len=8, 
                 hidden_dim=128, num_hidden_layers=0, 
                 kernel=nn.Linear, non_linear_kernel_pos='011',
                 skip_conn=True, skip_mode="concat",
                 inverse_norm=False, mean_norm=True,
                 chanel_independent=True, residual = True, verbose=False):
                 
        super(KUNet, self).__init__()
        n_enc_layers = len(n_width) + len(n_height) + 1
        if isinstance(hidden_dim, int):
           hidden_dim = [hidden_dim] * n_enc_layers
        if isinstance(num_hidden_layers, int):
           num_hidden_layers = [int(i) for i in non_linear_kernel_pos]
           kernel = [nn.Linear if i == "0" else kernel for i in non_linear_kernel_pos]

        self.model = KUNetEncoderDecoder(input_dim, input_len, n_width, n_height, 
                                        latent_dim, latent_len,
                                        output_dim, output_len, hidden_dim, 
                                        num_hidden_layers, kernel, non_linear_kernel_pos,
                                        skip_conn=skip_conn, skip_mode=skip_mode, verbose=verbose)
        self.inverse_norm = inverse_norm
        self.mean_norm = mean_norm
        self.chanel_independent = chanel_independent
        self.residual = residual

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
          output += res
        return output

    def set_latent_conditions(self, latent):
        self.model.latent_condition = latent

