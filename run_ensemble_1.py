import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import time

# checkpoint -> result 불러오기
import argparse
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
from utils.metrics import *
from utils.tools import EarlyStopping

from utils.metrics import *
from utils.tools import linear_regression_direct, linear_predict
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Custom

# 모델 훈련셋 결과 확인하기
from torch.utils.data import DataLoader

# 파서 불러오기
from commons.parser_write import parser

# fix random seed
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 스크립트 2개 정리 (./scripts/long_term_forecast/Multi_script/iTransformer_exchange_weather.sh)
scripts_list = ["""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_input_96 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --d_model 64\
  --d_ff 128\
  --des 'Exp' \
  --itr 1 """,

"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_input_192 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --batch_size 8 \
  --d_model 64\
  --d_ff 128\
  --des 'Exp' \
  --itr 1""",

]

args0 = parser.parse_args(scripts_list[0].split())
args0.use_gpu = True if torch.cuda.is_available() and args0.use_gpu else False

if args0.use_gpu and args0.use_multi_gpu:
    args0.devices = args0.devices.replace(' ', '')
    device_ids0 = args0.devices.split(',')
    args0.device_ids = [int(id_) for id_ in device_ids0]
    args0.gpu = args0.device_ids[0]


args1 = parser.parse_args(scripts_list[1].split())
args1.use_gpu = True if torch.cuda.is_available() and args1.use_gpu else False

if args1.use_gpu and args1.use_multi_gpu:
    args1.devices = args1.devices.replace(' ', '')
    device_ids1 = args1.devices.split(',')
    args1.device_ids = [int(id_) for id_ in device_ids1]
    args1.gpu = args1.device_ids[0]

# 스크립트 2개 정리 (./scripts/long_term_forecast/Multi_script/iTransformer_exchange_weather.sh)
exchange_96_96_result = "long_term_forecast_iTransformer_Exchange_96_96_Mod-iTransformer_data-exchange_rate.csv_(96to96)_0(1727353907)"
exchange_96_192_result = "long_term_forecast_iTransformer_input_192_Mod-iTransformer_data-exchange_rate.csv_(96to192)_0(1727354020)"

# 변경해야 할 부분
setting_pairs = [
    (exchange_96_96_result, args0),
    (exchange_96_192_result, args1),
]

idx = 0 # 순서
setting_path = setting_pairs[idx][0]
args = setting_pairs[idx][1]
args.gpu = 1

# 모델 호출 - Exp_Long_Term_Forecast - exchange_96_96
device = torch.device("cuda:1")
exp_model = Exp_Long_Term_Forecast(args)
exp_model.model.to(device)
exp_model._build_model()
# device = exp_model.device

# print(os.environ["CUDA_VISIBLE_DEVICES"], args.gpu, device)
# 위의 argument와 맞는 모델 호출
checkpoint_path = './checkpoints/'
model_path = f"{checkpoint_path}{setting_path}/checkpoint.pth"
model = torch.load(model_path, map_location="cuda:0")  # 0번 GPU로 매핑
exp_model.model.load_state_dict(model, strict=False)

# data_provider -> Exchange_rate
dataset_input = Dataset_Custom(args, args.root_path,
                    flag='train', size=(args.seq_len, args.label_len, args.pred_len),
                    features='M', data_path = args.data_path,
                    target='OT', scale=True, freq='h', timeenc=0,
                    seasonal_patterns=None, train_ratio=args.train_ratio, test_ratio=args.test_ratio)
dataset_input_test = Dataset_Custom(args, args.root_path,
                    flag='test', size=(args.seq_len, args.label_len, args.pred_len),
                    features='M', data_path = args.data_path,
                    target='OT', scale=True, freq='h', timeenc=0,
                    seasonal_patterns=None, train_ratio=args.train_ratio, test_ratio=args.test_ratio)

exp_model.model.eval()



dataset_input_loader = DataLoader(
            dataset_input,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False)
dataset_input_test_loader = DataLoader(
            dataset_input_test,
            batch_size=1, # 모든 데이터셋을 확인해야 해서 batch_size를 강제로 1로 조정.
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)

X = np.array([[t] for t in range(-args.seq_len, 0)])  # X는 입력 feature, shape: [seq_len, 1]
X_new = np.array([[t] for t in range(args.pred_len)])  # 예측을 위한 새로운 시간 변수
X_concat = np.concatenate([X, X_new], axis=0).reshape(-1)

# Combination 모델 제작
class CombinedModel(nn.Module):
    # 모델 정의 - 
    def __init__(self, res_A, res_B, res_C):
        super(CombinedModel, self).__init__()
        self.res_A = res_A  # iTransformer train_result
        self.res_B = res_B  # lin_reg_96 train_result
        self.res_C = res_C
        self.a = nn.Parameter(torch.ones(1, device=device)*0.998, requires_grad=True)
        self.b = nn.Parameter(torch.ones(1, device=device)*0.001, requires_grad=True)
        # self.c = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
        # self.d = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
    
    def forward(self, x):
        output_A = self.res_A(x)
        output_B = self.res_B(x)
        output_C = self.res_C(x)
        self.c = nn.Parameter(torch.ones(1, device=device), requires_grad=True) - self.a - self.b
        combined_output = self.a * output_A + self.b * output_B + output_C * self.c # + self.d
        return combined_output
    
    # model_output_function

def res_iTransformer(batch_x): # S 
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    return exp_model.model(batch_x, None, torch.zeros(B, len(X_new), N), None)

def res_lin_reg(batch_x):
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    # 각 배치와 변수에 대해 선형 회귀 해를 계산
    vals = [[linear_regression_direct(X, batch_x.permute(0,2,1)[idx, var , :], device) for var in range(N)] for idx in range(B)]
    lin_result = [[linear_predict(X_new, vals[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
    return lin_result

def res_lin_reg_24(batch_x):
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    # 24 조각에 대해서도 계산
    vals_24 = [[linear_regression_direct(X[-24:], batch_x.permute(0,2,1)[idx, var , -24:], device) for var in range(N)] for idx in range(B)]
    lin_result_24 = [[linear_predict(X_new, vals_24[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result_24 = torch.stack([torch.stack(lin_result_24[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
    return lin_result_24

def zero_model(batch_x): # S 길이
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    return torch.zeros(B, len(X_new), N)

# 우선 train_set의 data_exchange를 바탕으로 측정값 참값 가져기
# 트레인 데이터셋을 테스트해서 결과 받기, test 함수에서 가져옴
preds_te_tr = [] # 예측값
trues_te_tr = [] # 참값
preds_te_lin = [] # 96_lin
preds_te_lin_24 = [] # 24_lin

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_loader):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)

    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)

    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    # encoder - decoder

    # use_amp도 사용하지 않음, 
    outputs = exp_model.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # 각 배치와 변수에 대해 선형 회귀 해를 계산
    vals = [[linear_regression_direct(X, batch_x.permute(0,2,1)[idx, var , :], device) for var in range(N)] for idx in range(B)]
    lin_result = [[linear_predict(X_new, vals[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(device)

    # 24 조각에 대해서도 계산
    vals_24 = [[linear_regression_direct(X[-24:], batch_x.permute(0,2,1)[idx, var , -24:], device) for var in range(N)] for idx in range(B)]
    lin_result_24 = [[linear_predict(X_new, vals_24[idx][var], device) for var in range(N)] for idx in range(B)]
    # 결과를 3D 텐서로 변환
    lin_result_24 = torch.stack([torch.stack(lin_result_24[idx], dim=0) for idx in range(B)], dim=0).to(device)
    
    outputs = outputs[:, -args.pred_len:, :]
    batch_y = batch_y[:, -args.pred_len:, :].to(device)
    outputs = outputs.detach().cpu().numpy()
    batch_y = batch_y.detach().cpu().numpy()

    pred = outputs
    true = batch_y

    preds_te_tr.append(pred)
    trues_te_tr.append(true)
    preds_te_lin.append(lin_result)
    preds_te_lin_24.append(lin_result_24)

    if (i+1)%100==0:
        print(f"step {i+1} completed")
    
preds_te_tr = np.concatenate(preds_te_tr, axis=0)
trues_te_tr = np.concatenate(trues_te_tr, axis=0)
preds_te_lin = torch.concat(preds_te_lin, axis=0).detach().cpu().numpy()
preds_te_lin_24 = torch.concat(preds_te_lin_24, axis=0).detach().cpu().numpy()
preds_te_lin = np.transpose(preds_te_lin, (0,2,1))
preds_te_lin_24 = np.transpose(preds_te_lin_24, (0,2,1))


# 모델 실험
num_epochs = 5
combine_model_test = CombinedModel(res_iTransformer, res_lin_reg, res_lin_reg_24)
# combine_model_test training
combine_model_test.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam([combine_model_test.a, combine_model_test.b])


# 검증 데이터셋 결과 확인
def vali(vali_data, vali_loader, criterion):
    total_loss = []
    combine_model_test.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            targets = batch_y[:, -args.pred_len:, :].to(device)
            outputs = combine_model_test(batch_x)
            loss = criterion(outputs, targets)
            total_loss.append(loss)

    total_loss = [v.item() for v in total_loss]
    total_loss = np.average(total_loss)
    combine_model_test.train()
    return total_loss
# 모델 훈련

early_stopping = EarlyStopping(patience=2, verbose=True)
for epoch in range(num_epochs):
    cnt = 0
    train_loss = []
    # exp_model.train()
    path = os.path.join(args.checkpoints, setting_path)
    if not os.path.exists(path):
        os.makedirs(path)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_loader):
        cnt += 1
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        targets = batch_y[:, -args.pred_len:, :].to(device)
        optimizer.zero_grad()
        outputs = combine_model_test(batch_x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss)
        if (cnt+1) % 50 == 0:
            print(f"{cnt+1}th batch done, loss {loss}") 
    print("="*50)
    print(f"Epoch {epoch+1} DONE")
    print()
    train_loss = [v.item() for v in train_loss]
    train_loss = np.average(train_loss)
    vali_loss = vali(dataset_input_test, dataset_input_test_loader, criterion)
    print("vali_loss:", vali_loss)
    print()
    print(combine_model_test.a, combine_model_test.b)
    early_stopping(vali_loss, combine_model_test, path)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    model_path = path + '/' + 'checkpoint_ensenble.pth'
    # torch.save(combine_model_test.state_dict(), model_path)
    # combine_model_test.load_state_dict(torch.load(model_path, map_location=device))

# 훈련 결과 도출
combine_model_test.eval()
combine_model_test.a, combine_model_test.b, # combine_model_test.c # train 되지 않은 상태

# 기존 데이터의 메트릭 결과
MSE(preds_te_tr, trues_te_tr), MAE(preds_te_tr, trues_te_tr), SMAE(preds_te_tr, trues_te_tr)

# 계수 변경
a, b, = combine_model_test.a[0].item(), combine_model_test.b[0].item(),

# 새로운 모델 테트트
pred_combi = a*preds_te_tr + b*preds_te_lin + (1-a-b)*preds_te_lin_24
MSE(pred_combi, trues_te_tr), MAE(pred_combi, trues_te_tr), SMAE(pred_combi, trues_te_tr)

# 실제 데이터 셋 호출
result_list = ['pred.npy', 'true.npy']
result_path = './results/'
np_pred = np.load(f"{result_path}{setting_path}/{result_list[0]}")
np_true = np.load(f"{result_path}{setting_path}/{result_list[1]}")

# 이제 계산도 한다
# 각 배치와 변수에 대해 선형 회귀 해를 계산
B, L, N = np_pred.shape  # L은 시퀀스 길이(seq_len)
vals = [[linear_regression_direct(X, dataset_input_test[idx][0][:, var]) for var in range(N)] for idx in range(B)]
lin_result = [[linear_predict(X_new, vals[idx][var]) for var in range(N)] for idx in range(B)]
# 결과를 numpy 모듈로 변경
np_pred_lin = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1).detach().cpu().numpy()

vals2 = [[linear_regression_direct(X[-24:], dataset_input_test[idx][0][-24:, var], ) for var in range(N)] for idx in range(B)]
lin_result2 = [[linear_predict(X_new, vals2[idx][var]) for var in range(N)] for idx in range(B)]
# 결과를 numpy 모듈로 변경
np_pred_lin_24 = torch.stack([torch.stack(lin_result2[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1).detach().cpu().numpy()

# 마지막으로 비교
final_res = a*np_pred + b* np_pred_lin + (1-a-b)*np_pred_lin_24

# 메트릭 비교하기 (원본 iTransformer)
with open(f'run_ensenble_txt_{setting_path}_{time.time()}.txt', 'w', encoding='utf8') as A:
    wr = "TRAIN_PRED\n"
    wr += f"{MSE(np_pred, np_true), MAE(np_pred, np_true), SMAE(np_pred, np_true), REC_CORR(np_pred, np_true), STD_RATIO(np_pred, np_true), SLOPE_RATIO(np_pred, np_true)} \n"
    wr += "TRAIN_ENSEMBLE_PRED\n"
    wr += f"{MSE(final_res, np_true), MAE(final_res, np_true), SMAE(final_res, np_true), REC_CORR(final_res, np_true), STD_RATIO(final_res, np_true), SLOPE_RATIO(final_res, np_true)}\n"
    wr += "LIN_PRED\n"
    wr += "TRAIN_PRED_LINEAR\n"
    wr += f"{MSE(np_pred_lin, np_true), MAE(np_pred_lin, np_true), SMAE(np_pred_lin, np_true), REC_CORR(np_pred_lin, np_true), STD_RATIO(np_pred_lin, np_true), SLOPE_RATIO(np_pred_lin, np_true)}\n"
    wr += "TRAIN_PRED_LINEAR_24\n"
    wr += f"{MSE(np_pred_lin_24, np_true), MAE(np_pred_lin_24, np_true), SMAE(np_pred_lin_24, np_true), REC_CORR(np_pred_lin_24, np_true), STD_RATIO(np_pred_lin_24, np_true), SLOPE_RATIO(np_pred_lin_24, np_true)}\n"
    A.write(wr)

# 메트릭 저장
metric_path = f"./results/{setting_path}/"
np.save(metric_path + "pred_ensemble.npy", final_res)
np.save(metric_path + "pred_lin.npy", np_pred_lin)
np.save(metric_path + "pred_lin24.npy", np_pred_lin_24)
