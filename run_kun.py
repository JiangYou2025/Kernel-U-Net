import argparse
import os
import torch
from experiments import Exp_Main
import random
import numpy as np
import pickle as pkl

parser = argparse.ArgumentParser(description='KUN for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--train_only', action='store_true', default=False, help='train only desactivate the plotlines')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# KUN
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension in KUN, default 128')
parser.add_argument('--output_dim', type=int, default=128, help='output dimension in latent vector of KUN, default 128')
parser.add_argument('--input_dim', type=int, default=1, help='input dimension of patch, default 1')
parser.add_argument('--input_len', type=int, default=4, help='input dimension of patch, default 4')

parser.add_argument('--n_width', type=str, default="[1]", help='multiples of input dimension, default [1]')
parser.add_argument('--n_height', type=str, default="[5,6,6]", help='multiples of input length, default [5,6,6]')
parser.add_argument('--non_linear_kernel_pos', type=str, default="0000", help='position of non linear kernels, default 0000')
parser.add_argument('--non_linear_kernel', type=str, default="Linear", help='non linear kernels, default Linear')


parser.add_argument('--num_kun', type=int, default=1, help='num of stacked KUN, default 1')
parser.add_argument('--tau_earlystopping', type=float, default=0.9, help='tau for weighted earlystopping, tau \in [0, 1]')
parser.add_argument('--use_random_erase', action='store_true', default=False, help='use random erase in transforms; True 1 False 0')
parser.add_argument('--use_chanel_independence', action='store_true', default=False, help='use chanel independence setting; True 1 False 0')
parser.add_argument('--use_unet_skip', action='store_true', default=False, help='use unet skip at each layers; True 1 False 0')
parser.add_argument('--use_instance_norm', action='store_true', default=False, help='use instance norm, or mean norm; True 1 False 0')
    
parser.add_argument('--use_pickle_log', action='store_true', default=False, help='use pickle log')
parser.add_argument('--pkl_log_name', type=str, default="pickle.pkl", help=' pickle log file name')
parser.add_argument('--reset', action='store_true', default=False, help='reset pickle log file')

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')


# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', default=False, help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--criterion', type=str, default='MSE', help='loss function')
parser.add_argument('--custom_sampler', type=str, default='None', help='InfoBatch, None, ...')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')

parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--dp_rank', type=int,default = 8)
parser.add_argument('--rescale', type=int,default = 1)
parser.add_argument('--merge_size',type=int,default = 2)
parser.add_argument('--momentum', type=float,default = 0.1)
parser.add_argument('--local_rank', type=int,default = 0)
parser.add_argument('--devices_number',type=int,default = 1)
parser.add_argument('--use_statistic',action='store_true', default=False)
parser.add_argument('--use_decomp',action='store_true', default=False)
parser.add_argument('--same_smoothing',action='store_true', default=False)
parser.add_argument('--warmup_epochs',type=int,default = 0)

# GPU
parser.add_argument('--use_gpu', action='store_true', default=False, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--device', type=str, default='cpu', help='device name of cuda:0 or cpu')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)


args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu :
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    args.device="cuda:0"
elif args.use_gpu:
    args.device="cuda:0"
else :
    args.device="cpu"

args.n_width = list([int(i)  for i in args.n_width.replace("[", "" ).replace("]", "" ).replace(",", "" )])
args.n_height = list([int(i)  for i in args.n_height.replace("[", "" ).replace("]", "" ).replace(",", "" )])
args.portion = [1, 1]

args.hidden_dim = [args.hidden_dim] * len(args.non_linear_kernel_pos) if isinstance(args.hidden_dim, int) else args.hidden_dim
args.num_hidden_layers = [int(i) for i in args.non_linear_kernel_pos]
args.kernal_model = []
for  i in args.non_linear_kernel_pos:
    if int(i) == 0:
      args.kernal_model.append("Linear")
    else:
     args.kernal_model.append(args.non_linear_kernel)

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    result = []
    start_run = 0
    experiment_name = f'Experiment_{args.non_linear_kernel}_{args.non_linear_kernel_pos}_{args.custom_sampler}_'+ \
                      f"{args.data}_{args.model}_{args.features}_{args.seq_len}_{args.pred_len}_"#+ \
                      #f"mean_bias_{mean_bias}_sigma_{sigma}_"

    if args.use_pickle_log:
        if not os.path.exists(args.pkl_log_name):
            with open(args.pkl_log_name, 'wb') as f:
                pkl.dump({}, f)
        with open(args.pkl_log_name, "rb") as f:
            result_dict = pkl.load(f)

        start_run = 0
        if experiment_name in result_dict.keys() and not args.reset:
            start_run = len(result_dict[experiment_name])
            if start_run >=args.itr:
                print("skip ", experiment_name)
                exit(0) 
            else:
                result = result_dict[experiment_name]

    for ii in range(start_run, args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, mse = exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()

        if args.use_pickle_log:

            test_mse_list = exp.test_mse_list
            train_mse_list = exp.train_mse_list
            val_mse_list = exp.val_mse_list

            top_5_mse = sorted(test_mse_list)[:5]
            top_5_mse_mean, top_5_mse_std = np.mean(top_5_mse), np.std(top_5_mse)

            result.append([mae, mse, top_5_mse_mean, top_5_mse_std, train_mse_list, val_mse_list, test_mse_list, args])

            with open(args.pkl_log_name, "rb") as f:
                result_dict = pkl.load(f)
            result_dict.update({experiment_name:result})
            with open(args.pkl_log_name, "wb") as f:
                pkl.dump(result_dict, f)

else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()