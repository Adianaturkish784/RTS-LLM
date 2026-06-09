import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, RTSLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='RTS-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, choices=[0, 1], default=1,
                    help='1: train, 0: test only')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='checkpoint file used when --is_training 0')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# Self-supervised parameter
parser.add_argument('--W_rebuild',type=float,default=0.2,help='the weight of rebuild_loss')
parser.add_argument('--mask_rate',type=float,default=0.125,help='the rate of mask')
parser.add_argument('--W_nsp',type=float,default=0.2,help='the weight of nsp_loss')
parser.add_argument('--pair_num',type=int,default=30,help='the pair num of nsp for every input')


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='half_decay', help='learning rate adjustment strategy; one_cycle adjusts the learning rate in each batch')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--test_batches', type=int, default=0,
                    help='maximum test batches when --is_training 0; 0 evaluates all batches')

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
if args.is_training:
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
else:
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}_lr{}_{}_{}'.format(
        args.task_name,
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
        args.des,
        ii,
        args.learning_rate,
        args.lradj,
        args.llm_model
    )

    args.content = load_content(args)
    test_data, test_loader = data_provider(args, 'test')
    if args.is_training:
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = RTSLLM.Model(args).float()

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    if not args.is_training:
        if not args.checkpoint_path:
            raise ValueError('--checkpoint_path is required when --is_training 0')
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint_path}')

        accelerator.print(f'Loading checkpoint: {args.checkpoint_path}')
        try:
            state_dict = torch.load(args.checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

        if hasattr(model, 'llm_model'):
            model.llm_model.config.output_attentions = False
            model.llm_model.config.output_hidden_states = False

        model, test_loader = accelerator.prepare(model, test_loader)
        test_loss, test_mae_loss = vali(
            args,
            accelerator,
            model,
            test_data,
            test_loader,
            criterion,
            mae_metric,
            max_batches=args.test_batches or None,
        )
        accelerator.print(f'Test MSE: {test_loss:.7f} | Test MAE: {test_mae_loss:.7f}')
        continue

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    mse,mae = None,None
    gradient_explosion = False
    record_path = "result_"+args.task_name+".txt"
    f = open(record_path, 'a')
    f.write(setting + "  \n")
    f.write(str(args) + "  \n")
    f.write("lr:"+str(args.learning_rate)+ "  \n")
    f.close()
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader)):
            if len(batch) == 8:
                batch_x, batch_y, batch_x_mark, batch_y_mark, mask, seq_start, pred_start, nsp_label = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                mask, seq_start, pred_start, nsp_label = None, None, None, None

            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            outputs, rebuild_loss, nsp_loss = model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                mask=mask, seq_start=seq_start, pred_start=pred_start, nsp_label=nsp_label
            )
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            if loss.item() > 10 :
                gradient_explosion = True
            pred_loss = loss
            train_loss.append(pred_loss.item())
            loss += args.W_rebuild*rebuild_loss
            loss += args.W_nsp*nsp_loss

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f} | rebuildloss: {3:.7f} | nsploss: {4:.7f}"
                    .format(i + 1, epoch + 1, pred_loss.item(), rebuild_loss.item(), nsp_loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            accelerator.backward(loss)
            model_optim.step()

            if args.lradj == 'one_cycle':
                if (i + 1) % 100 == 0:
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
                else:
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        # log
        with open(record_path, 'a') as f:
            f.write("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}\n".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))
            if early_stopping.counter != 0:
                f.write(f'EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}\n')

        if early_stopping.counter == 0:
            mse,mae= test_loss,test_mae_loss
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'one_cycle':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
    f = open(record_path, 'a')
    f.write('Gradient explosion status:{}\n'.format(gradient_explosion))
    f.write('Result: mse:{}, mae:{}\n\n'.format(mse, mae))
    f.close()
if accelerator.num_processes > 1:
    accelerator.wait_for_everyone()
