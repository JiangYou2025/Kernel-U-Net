import pickle as pkl
from sklearn.preprocessing import StandardScaler
import itertools

import numpy as np
import pandas as pd

import os
import time

import warnings
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim

from models import NKUNet
from dataset_provider import *

from torchvision.transforms import RandomRotation, RandomAffine, RandomErasing, RandomGrayscale, RandomApply, RandomHorizontalFlip,RandomAdjustSharpness,RandomCrop
from torchvision.transforms.v2 import ColorJitter


warnings.filterwarnings('ignore')

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.995 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            20:0.00008, 50: 0.00006,  80: 0.00004,
        }
    elif args.lradj == 'type3':
        lr_adjust = {
            10:0.00008, 20:0.00006, 50: 0.00004,  80: 0.00001,
        }
    elif args.lradj == 'type4':
        lr_adjust = {
            1:0.0001, 20:0.00008, 50: 0.00006,  80: 0.00004,
        }
    elif args.lradj == 'type5':
        lr_adjust = {
            1:0.0001, 10:0.00005, 20: 0.00003,  30: 0.00001, 40: 0.000005, 50: 0.000001,
        }
    elif args.lradj == 'type6':
        lr_adjust = {
            1:0.0001, 10:0.00005, 20: 0.00003,  30: 0.00001
        }
    elif args.lradj == 'type7':
        lr_adjust = {
            1:0.0001, 10:0.00005, 20: 0.00004,  30: 0.00003
        }
    elif args.lradj == 'type8':
        lr_adjust = {
            1:0.001, 10:0.0005, 20: 0.0003,
        }
    elif args.lradj == 'type9':
        lr_adjust = {
            1:0.00005, 10:0.00001, 30: 0.000005
        }
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, tau=0.5):
        new_val_loss = tau * self.val_loss_min + (1 - tau) * val_loss if self.val_loss_min is not np.Inf else val_loss
        score = -new_val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(new_val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(new_val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.show()

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.train_mse_list = []
        self.test_mse_list = []
        self.val_mse_list = []
        self.time_cost_list = []

        self.args = args
        self.transforms = RandomErasing(p=0.3, scale=(0.02, 0.23)).to(self.args.device)

        self.MSELoss = nn.MSELoss(reduce=False)

    def _build_model(self):
        model_dict = {
            #'Autoformer': Autoformer,
            #'Transformer': Transformer,
            #'CARDModelLinear': CARDModel,
            #'PatchTSTLinear': PatchTST,
            'NKUNet': NKUNet,
        }
        model = model_dict[self.args.model](self.args).float().to(self.device)
        print(model)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.criterion == 'MSELoss':
          criterion = nn.MSELoss(reduce=False)
        elif self.args.criterion == 'L1Loss':
          criterion = nn.L1Loss(reduce=False)
        elif self.args.criterion == 'L1Loss+MSELoss':
          criterion = nn.L1Loss(reduce=False)
        else:
          criterion = nn.MSELoss(reduce=False)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x
                batch_y = batch_y

                dec_inp= None
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x) 
                    elif 'KUNet' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true).mean()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        self.train_mse_list = []
        self.test_mse_list = []
        self.val_mse_list = []
        self.cost_time_list = []
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_loader.sampler.reset()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            if self.args.use_unet_skip:
              self.model.linear_list[0].use_unet_skip = self.args.use_unet_skip

            epoch_time = time.time()
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x

                if self.args.use_random_erase:
                  batch_x = self.transforms(batch_x)
                batch_y = batch_y

                dec_inp= None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss_no_reduction = criterion(outputs, batch_y)
                        loss = loss_no_reduction.mean()
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    
                    elif 'KUNet' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss_no_reduction = criterion(outputs, batch_y)
                    loss = loss_no_reduction.mean()
                    train_loss.append(loss.item())

                    if self.args.custom_sampler == "gaussian":
                      train_loader.sampler.append_idx_loss(index, loss_no_reduction.mean((1,2)).detach().cpu())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            cost_time = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, cost_time))
            self.cost_time_list.append(cost_time)
            train_loss = np.average(train_loss)
            self.train_mse_list.append(train_loss)
            if not self.args.train_only:
                if True or epoch % 10 == 0:
                  vali_loss = self.vali(vali_data, vali_loader, criterion=self.MSELoss)
                  test_loss = self.vali(test_data, test_loader, criterion=self.MSELoss)
                  self.test_mse_list.append(test_loss)
                  self.val_mse_list.append(vali_loss)

                  print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                      epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                  early_stopping(vali_loss, self.model, path, tau = self.args.tau_earlystopping)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path, tau = self.args.tau_earlystopping)
            #print("contrast_error", contrast_error.item())
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.custom_sampler == "gaussian":
                train_loader.sampler.update_weights(tau=0.8, min_weight=1, max_weight=self.args.max_weight, gaussian=True,
                                                mean_bias=self.args.mean_bias, sigma=self.args.sigma, )
                
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x
                batch_y = batch_y

                batch_x_mark = batch_x_mark
                batch_y_mark = batch_y_mark
                dec_inp= None

                # decoder input
                #dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    elif 'KUNet' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if False and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        #folder_path = './results/' + setting + '/'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        #f = open("result.txt", 'a')
        #f.write(setting + "  \n")
        #f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        #f.write('\n')
        #f.write('\n')
        #f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        self.model.train()
        return mae, mse 

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def search_architecture(
    non_linear_kernel,
    f_name,
    data = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
    seq_len = [336, 720],
    pred_len = [96, 192, 336, 720],
    non_linear_kernel_pos = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111',
                             '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'],
    root_path = path + '/ETT-small/',
    features= 'M',
    patience=50,
    train_epochs=50,
    lradj='type1',
    learning_rate=0.00005,
    hidden_dim = 128,
    tau_earlystopping = 0.9,
    use_random_erase = True,
    use_chanel_independence = True,
    use_unet_skip = True,
    use_instance_norm = False,
    num_runs = 1,
    portion=[1,1],

    custom_sampler = "info_batch",
    ib_ratio = 0.3,
    ib_delta = 0.875,
    max_weight = 100,
    mean_bias = 0.0,
    sigma=1.0,
    reset = False,
    ):

  iters = itertools.product(data, seq_len, pred_len, non_linear_kernel_pos)

  with open(f_name, "rb") as f:
    result_dict = pkl.load(f)

  for data, seq_len, pred_len, non_linear_kernel_pos in iters:
    print("process..", data, seq_len, pred_len, non_linear_kernel_pos, mean_bias, sigma)

    class Args():
      def __init__(self):
        self.data= data
        self.embed= 0 #'timeF'
        self.train_only= False
        self.batch_size= 128
        self.freq= 'h'
        self.root_path= root_path
        self.data_path= data + '.csv'
        if data == "Traffic":
          self.data = data
          self.root_path= path + '/traffic/'
          self.data_path= 'traffic.txt'
        if data == "Weather":
          self.data = data
          self.root_path= path + '/weather_max_planck/'
          self.data_path= 'weather.csv'
        if data == "Electricity":
          self.data = data
          self.root_path= path + '/electricity/'
          self.data_path= 'electricity.txt'
        if data == "ILI":
          self.data = data
          self.root_path= path + '/ili/'
          self.data_path= 'ILINet_v3.csv'


        self.seq_len= seq_len
        self.label_len= seq_len
        self.pred_len= pred_len
        self.features= features #MS,"M", "S"
        self.target = 'OT' # target column name
        self.num_workers=0

        self.use_gpu=True
        self.gpu=0
        self.use_multi_gpu=False

        self.model="KUN"
        self.enc_in=7
        self.individual=True
        self.checkpoints="checkpoints"
        self.patience=patience

        self.learning_rate=learning_rate
        self.use_amp=False
        self.train_epochs=train_epochs

        self.lradj=lradj
        self.test_flop=False

        self.output_attention=False

        self.portion=portion

        self.tau_earlystopping = tau_earlystopping
        self.use_random_erase = use_random_erase
        self.use_unet_skip = use_unet_skip
        self.use_chanel_independence = use_chanel_independence
        self.use_instance_norm = use_instance_norm

        self.custom_sampler = custom_sampler
        self.ib_ratio = ib_ratio
        self.ib_delta = ib_delta

        ###
        self.max_weight = max_weight
        self.sigma = sigma
        self.mean_bias = mean_bias
        ###


        self.output_dim = hidden_dim
        self.hidden_dim = [hidden_dim]*6
        self.input_dim = 1
        self.input_len = 4
        self.n_width = [1]
        if seq_len == 336 and pred_len<=336:
          self.n_height = [7, 3, 4]
        elif seq_len == 336 and pred_len==720:
          self.n_height = [[7, 3, 4], [5,6,6]]
        elif seq_len == 720:
          self.n_height = [5,6,6]
        elif seq_len == 144:
          self.n_height = [4,3,3]
        elif seq_len == 128:
          self.n_height = [2,4,4]

        self.num_kun = 1
        self.num_hidden_layers = [int(i) for i in non_linear_kernel_pos]
        self.kernal_model = []
        for  i in non_linear_kernel_pos:
          if int(i) == 0:
            self.kernal_model.append(nn.Linear)
          else:
            self.kernal_model.append(non_linear_kernel)

    args = Args()
    result = []
    params = f"patience_{patience}_train_epochs_{train_epochs}_"
    params += f"lradj_{lradj}_learning_rate_{learning_rate}_"
    params += f"hidden_dim_{hidden_dim}_hidden_dim_{hidden_dim}_"
    params += f"tau_earlystopping_{tau_earlystopping}_custom_sampler_{custom_sampler}_"
    params += f"mean_bias_{mean_bias}_sigma{sigma}_"

    experiment_name = f'Experiment_{non_linear_kernel}_{non_linear_kernel_pos}_{custom_sampler}_'+ \
                      f"{args.data}_{args.model}_{args.features}_{args.seq_len}_{args.pred_len}_"#+ \
                      #f"mean_bias_{mean_bias}_sigma_{sigma}_"
    start_run = 0
    if experiment_name in result_dict.keys() and not reset:
      start_run = len(result_dict[experiment_name])
      if start_run >=num_runs:
        print("skip ", experiment_name)
        continue
      else:
        result = result_dict[experiment_name]

    for i in range (start_run, num_runs):


      expm = Exp_Main(args)
      expm.train(setting=f"{args.data}_{args.model}_{non_linear_kernel}_{non_linear_kernel_pos}_{args.features}_{args.seq_len}_{args.pred_len}")
      mae, mse = expm.test(setting=f"{args.data}_{args.model}_{non_linear_kernel}_{non_linear_kernel_pos}_{args.features}_{args.seq_len}_{args.pred_len}", test=1)


      test_mse_list = expm.test_mse_list
      train_mse_list = expm.train_mse_list
      val_mse_list = expm.val_mse_list

      top_5_mse = sorted(test_mse_list)[:5]
      top_5_mse_mean, top_5_mse_std = np.mean(top_5_mse), np.std(top_5_mse)

      result.append([mae, mse, top_5_mse_mean, top_5_mse_std, train_mse_list, val_mse_list, test_mse_list, params])

      with open(f_name, "rb") as f:
        result_dict = pkl.load(f)
      result_dict.update({experiment_name:result})
      with open(f_name, "wb") as f:
        pkl.dump(result_dict, f)
