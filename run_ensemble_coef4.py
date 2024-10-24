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
from commons.parser_write import *

# fix random seed
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 스크립트 8개 정리 (./scripts/long_term_forecast/Multi_script/iTransformer_exchange_weather.sh)


scripts_list = [
"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_96 \
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
  --itr 1""", 

"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_192 \
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
  
"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_336 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
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


"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_720 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
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

"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id iTransformer_weather_96_96 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 64\
  --d_ff 128\
  --itr 1 """,

"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id iTransformer_weather_96_192 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 64\
  --d_ff 128\
  --itr 1 """,
  
"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id iTransformer_weather_96_336 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 64\
  --d_ff 128\
  --itr 1 """,
  
"""--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id iTransformer_weather_96_720 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --batch_size 32 \
  --d_model 64\
  --d_ff 128\
  --itr 1 
"""]

args_list = [] # argument list

for c in range(8):
    arg = parser.parse_args(scripts_list[c].split())
    arg.use_gpu = True if torch.cuda.is_available() and arg.use_gpu else False
    
    if arg.use_gpu and arg.use_multi_gpu:
        arg.devices = arg.devices.replace(' ', '')
        device_ids = arg.devices.split(',')
        arg.device_ids = [int(id_) for id_ in device_ids]
        arg.gpu = arg.device_ids[0]
    
    args_list.append(arg)

# 스크립트 8개 정리 (./scripts/long_term_forecast/Multi_script/iTransformer_exchange_weather.sh)
exchange_96_96_result = "long_term_forecast_iTransformer_Exchange_96_96_Mod-iTransformer_data-exchange_rate.csv_(96to96)_0(1727614375)"
exchange_96_192_result = "long_term_forecast_iTransformer_Exchange_96_192_Mod-iTransformer_data-exchange_rate.csv_(96to192)_0(1727520705)"
exchange_96_336_result = "long_term_forecast_iTransformer_Exchange_96_336_Mod-iTransformer_data-exchange_rate.csv_(96to336)_0(1727521844)"
exchange_96_720_result = "long_term_forecast_iTransformer_Exchange_96_720_Mod-iTransformer_data-exchange_rate.csv_(96to720)_0(1727520705)"
weather_96_96_result = "long_term_forecast_iTransformer_weather_96_96_Mod-iTransformer_data-weather.csv_(96to96)_0(1727354116)"
weather_96_192_result = "long_term_forecast_iTransformer_weather_96_192_Mod-iTransformer_data-weather.csv_(96to192)_0(1727354589)"
weather_96_336_result = "long_term_forecast_iTransformer_weather_96_336_Mod-iTransformer_data-weather.csv_(96to336)_0(1727355118)"
weather_96_720_result = "long_term_forecast_iTransformer_weather_96_720_Mod-iTransformer_data-weather.csv_(96to720)_0(1727355677)"

# 변경해야 할 부분
setting_pairs = [
    (exchange_96_96_result, args_list[0]),
    (exchange_96_192_result, args_list[1]),
    (exchange_96_336_result, args_list[2]),
    (exchange_96_720_result, args_list[3]),
    (weather_96_96_result, args_list[4]),
    (weather_96_192_result, args_list[5]),
    (weather_96_336_result, args_list[6]),
    (weather_96_720_result, args_list[7])
]

# 역함수
def sigmoid_inverse(y):
    # y는 0과 1 사이의 값이어야 합니다.
    return np.log(y / (1 - y))

idx = 0 # 순서
col_count = 6 # 한 에포크당 수집 데이터 수
num_epochs = 5 # 에포크 ㅅ횟수
use_gpu = 4 # 사용 GPU 번호 - 오류 잡기 위해 
# q1, q2 = "lin96", "lin24" # 앙상블 모델 텍스트
q1, q2 = "lin96", "none" # 앙상블 모델 텍스트
a_init , b_init = sigmoid_inverse(0.999), sigmoid_inverse(0.001)  # 초기값(sigmoid로변환할  것 감안)  
lr = 0.01 #gradient descending 속도



# 시작값 기준
pair_settings = [
  {"q1":"lin96", "q2": "none"},
  {"q1":"lin48", "q2": "none"},
  {"q1":"lin24", "q2": "none"},
  {"q1":"lin12", "q2": "none"},
]

for pair_map in pair_settings:
    q1, q2 = pair_map["q1"], pair_map["q2"]
    
    setting_path = setting_pairs[idx][0]
    args = setting_pairs[idx][1]
    args.gpu = use_gpu
    
    # 모델 호출 - Exp_Long_Term_Forecast - exchange_96_96
    exp_model = Exp_Long_Term_Forecast(args)
    exp_model._build_model()
    device = torch.device(f"cuda:{use_gpu}")
    exp_model.model.device = device
    # device = exp_model.device
    
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
    
    # Combination 모델 제작, 2단계/3단계 대응
    class CombinedModel(nn.Module):
        # 모델 정의 - 
        def __init__(self, res_A, res_B, res_C):
            super(CombinedModel, self).__init__()
            self.res_A = res_A  # iTransformer train_result
            self.res_B = res_B  # lin_reg_96 train_result
            self.res_C = res_C
            self.a = nn.Parameter(torch.ones(1, device=device)*a_init, requires_grad=True)
            self.b = nn.Parameter(torch.ones(1, device=device)*b_init, requires_grad=True)
            if res_C is not None:
                self.c = nn.Parameter(torch.ones(1, device=device)*(1-a_init -b_init), requires_grad=True)
            else:
                self.c = nn.Parameter(torch.ones(1, device=device)*0.0, requires_grad=True)
            # self.c = nn.Parameter(torch.ones(1, device=device)*0.0, requires_grad=True)
            # self.d = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
            self.a_sigmoid = torch.sigmoid(self.a)
            self.b_sigmoid = torch.sigmoid(self.b)
        
        def set_a(self, val):
            # nn.Parameter를 다시 생성하지 않고 값을 설정
            with torch.no_grad():
                self.a.copy_(torch.tensor([val], device=device))
    
        def set_b(self, val):
            # nn.Parameter를 다시 생성하지 않고 값을 설정
            with torch.no_grad():
                self.b.copy_(torch.tensor([val], device=device))
                
        
        """def forward(self, x):
            output_A = self.res_A(x)
            output_B = self.res_B(x)
            
            # 계수를 0 이상으로 제한
            with torch.no_grad():
                # self.a.data = torch.nn.functional.softplus(self.a)  # Softplus 적용
                self.a.data = torch.clamp(self.a.data, min=0)

            if self.res_C is None:
                with torch.no_grad():
                    self.b.data = torch.clamp(self.b.data, min=0)
                    self.b.copy_(torch.ones(1, device=device) - self.a)
                    
                combined_output = self.a * output_A + self.b * output_B
            else:
                output_C = self.res_C(x)
                with torch.no_grad():
                    # self.b.data = torch.nn.functional.softplus(self.b)  # Softplus 적용
                    self.b.data = torch.clamp(self.b.data, min=0)
                    self.c.data = torch.clamp(self.c.data, min=0)
                    self.c.copy_(torch.ones(1, device=device) - self.a - self.b)
                combined_output = self.a * output_A + self.b * output_B + output_C * self.c
            
            return combined_output"""
        def forward(self, x):
            output_A = self.res_A(x)
            output_B = self.res_B(x)
        
            # Apply transformations to ensure non-negative parameters
            a_sigmoid = torch.sigmoid(self.a)
            
            # Ensure the sum constraint by defining c as the remainder
            b_sigmoid = torch.clamp(1 - a_sigmoid, min=0)
        
            # Compute the combined output with updated `a`, `b`, `c`
            combined_output = a_sigmoid * output_A + b_sigmoid * output_B
            
            self.a_sigmoid = a_sigmoid
            self.b_sigmoid = b_sigmoid
        
            """
            if self.res_C is not None:
                b_pos = torch.nn.functional.softplus(self.b)
                combined_output = a_pos * output_A + b_pos * output_B
                # Ensure the sum constraint by calculating `c_pos`
                c_pos = torch.clamp(torch.ones(1, device=self.a.device) - a_pos - b_pos, min=0)
                output_C = self.res_C(x)
                combined_output += c_pos * output_C
            """    
        
            return combined_output
        
        def get_result(self):
            a, b = self.a_sigmoid.detach().cpu().numpy()[0], self.b_sigmoid.detach().cpu().numpy()[0]
            return a,b
        
    # model_output_function
    
    def res_iTransformer(batch_x): # S 
        B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
        return exp_model.model(batch_x, None, torch.zeros(B, len(X_new), N), None)
    
    def res_lin_reg(batch_x, reg_size=96):
        B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
        # 각 배치와 변수에 대해 선형 회귀 해를 계산
        vals = [[linear_regression_direct(X[-reg_size:], batch_x.permute(0,2,1)[idx, var , -reg_size:], device) for var in range(N)] for idx in range(B)]
        lin_result = [[linear_predict(X_new, vals[idx][var], device) for var in range(N)] for idx in range(B)]
        # 결과를 3D 텐서로 변환
        lin_result = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
        return lin_result
    
    def zero_model(batch_x): # S 길이
        B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
        return torch.zeros(B, len(X_new), N)
    
    # 함수 도출
    def get_res_lin(reg_size):
        def fn(reg_size):
            return res_lin_reg(reg_size)
        
        return fn
        
    def res_lin_reg_24(batch_x):
        B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
        # 24 조각에 대해서도 계산
        vals_24 = [[linear_regression_direct(X[-24:], batch_x.permute(0,2,1)[idx, var , -24:], device) for var in range(N)] for idx in range(B)]
        lin_result_24 = [[linear_predict(X_new, vals_24[idx][var], device) for var in range(N)] for idx in range(B)]
        # 결과를 3D 텐서로 변환
        lin_result_24 = torch.stack([torch.stack(lin_result_24[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
        return lin_result_24
    
    
    def res_lin_reg_48(batch_x):
        B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
        # 48 조각에 대해서도 계산
        vals_48 = [[linear_regression_direct(X[-48:], batch_x.permute(0,2,1)[idx, var , -48:], device) for var in range(N)] for idx in range(B)]
        lin_result_48 = [[linear_predict(X_new, vals_48[idx][var], device) for var in range(N)] for idx in range(B)]
        # 결과를 3D 텐서로 변환
        lin_result_48 = torch.stack([torch.stack(lin_result_48[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
        return lin_result_48
    
    def res_lin_reg_12(batch_x):
        B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
        # 12 조각에 대해서도 계산
        vals_12 = [[linear_regression_direct(X[-12:], batch_x.permute(0,2,1)[idx, var , -12:], device) for var in range(N)] for idx in range(B)]
        lin_result_12 = [[linear_predict(X_new, vals_12[idx][var], device) for var in range(N)] for idx in range(B)]
        # 결과를 3D 텐서로 변환
        lin_result_12 = torch.stack([torch.stack(lin_result_12[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1)
        return lin_result_12
    
    QMAP_FN = {
      "none": None,
      "lin12" : res_lin_reg_12,
      "lin24" : res_lin_reg_24,
      "lin48" : res_lin_reg_48,
      "lin96" : res_lin_reg,
      "lin" : res_lin_reg
    }

    
         
    # 우선 train_set의 data_exchange를 바탕으로 측정값 참값 가져기
    # 트레인 데이터셋을 테스트해서 결과 받기, test 함수에서 가져옴
    preds_te_tr = [] # 예측값
    trues_te_tr = [] # 참값
    preds_te_lin = [] # 96_lin
    preds_te_lin_48 = [] # 48_lin
    preds_te_lin_24 = [] # 24_lin
    preds_te_lin_12 = [] # 24_lin
    
    start_time = time.time() # 시작타
    print(f"iteration start, time : {time.localtime()}")
    
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
        lin_result = res_lin_reg(batch_x, 96).to(device)
    
        lin_result_24 = res_lin_reg(batch_x, 24).to(device)
        
        # 48 테스트
        lin_result_48 = res_lin_reg(batch_x, 48).to(device)
        
        # 12 테스트
        lin_result_12 = res_lin_reg(batch_x, 12).to(device)
        
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
        preds_te_lin_48.append(lin_result_48)
        preds_te_lin_12.append(lin_result_12)
    
        if (i+1)%100==0:
            print(f"step {i+1} completed")
    
        
    preds_te_tr = np.concatenate(preds_te_tr, axis=0)
    trues_te_tr = np.concatenate(trues_te_tr, axis=0)
    preds_te_lin = np.transpose(torch.concat(preds_te_lin, axis=0).detach().cpu().numpy(), (0,2,1))
    preds_te_lin_24 = np.transpose(torch.concat(preds_te_lin_24, axis=0).detach().cpu().numpy(), (0,2,1))
    preds_te_lin_48 = np.transpose(torch.concat(preds_te_lin_48, axis=0).detach().cpu().numpy(), (0,2,1))
    preds_te_lin_12 = np.transpose(torch.concat(preds_te_lin_12, axis=0).detach().cpu().numpy(), (0,2,1))
    input_len = i + 1
    
    # operation 끝
    data_col_time = time.time()
    print(f"경과 시간: {data_col_time-start_time}")
    
    # 모델 실험
    combine_model_test = CombinedModel(res_iTransformer, QMAP_FN[q1], QMAP_FN[q2])
    # combine_model_test.set_a(0.95)
    # combine_model_test.set_b(0.025)
    # combine_model_test training
    combine_model_test.train()
    # torch.nn.utils.clip_grad_norm_(combine_model_test.parameters(), max_norm=1.0) # gradient clipping - 크기 제한
    
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(combine_model_test.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam([combine_model_test.a], lr=lr, weight_decay=1e-3)
    
    # 검증 데이터셋 결과 확인
    def vali(vali_data, vali_loader, criterion):
        total_loss = []
        combine_model_test.eval()
        len_data = (len(vali_data)-1)//5 
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
    
    loss_points = [] # (a, b)
    # input_len = int(np.ceil(len(dataset_input) / args.batch_size) )
    input_len_div = int(np.ceil(input_len / (col_count - 1)))
    
    print("INPUT_LEN", input_len, input_len_div)
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
            if i == input_len -1 or i % input_len_div == 0:
                a, b = torch.sigmoid(combine_model_test.a).detach().cpu().numpy()[0], 1 - torch.sigmoid(combine_model_test.a).detach().cpu().numpy()[0]
                print(f"STEP {i} , {a,b}", "FIX", combine_model_test.get_result() )
                loss_points.append((a,b))
                vali_loss = vali(dataset_input_test, dataset_input_test_loader, criterion)
                print("vali_loss:", vali_loss)
                
        print("="*50)
        print(f"Epoch {epoch+1} DONE")
        print()
        train_loss = [v.item() for v in train_loss]
        train_loss = np.average(train_loss)
        vali_loss = vali(dataset_input_test, dataset_input_test_loader, criterion)
        # print("vali_loss:", vali_loss)
        # print()
        print(a,b)
        # early_stopping(vali_loss, combine_model_test, path)
        # if early_stopping.early_stop:
        #    print("Early stopping")
        #    break
        model_path = path + '/' + 'checkpoint_ensenble.pth'
        # torch.save(combine_model_test.state_dict(), model_path)
        # combine_model_test.load_state_dict(torch.load(model_path, map_location=device))
    
    # 훈련 결과 도출
    combine_model_test.eval()
    
    # 계수 수집 끝
    end_time = time.time()
    print(f"Find Coefficients process done. Spent :{end_time - start_time}")

    
    # 실제 데이터 셋 호출
    result_list = ['pred.npy', 'true.npy']
    result_path = './results/'
    np_pred = np.load(f"{result_path}{setting_path}/{result_list[0]}")
    np_true = np.load(f"{result_path}{setting_path}/{result_list[1]}")
    
    # 선형회귀 계수 결과값으로 계산
    def get_np_pred_lin(np_pred, reg_size=96):
        # 이제 계산도 한다
        # 각 배치와 변수에 대해 선형 회귀 해를 계산
        B, L, N = np_pred.shape  # L은 시퀀스 길이(seq_len)
        vals = [[linear_regression_direct(X[-reg_size:], dataset_input_test[idx][0][-reg_size:, var]) for var in range(N)] for idx in range(B)]
        lin_result = [[linear_predict(X_new, vals[idx][var]) for var in range(N)] for idx in range(B)]
        # 결과를 numpy 모듈로 변경
        np_pred_lin = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(device).permute(0,2,1).detach().cpu().numpy()
        return np_pred_lin
    
    
    # q1, q2가 잇을 때 
    
    if q1 == "lin96":
        np_pred_first = get_np_pred_lin(np_pred, 96)
        print("PRED_96_DONE")
    elif q1 == "lin48":
        np_pred_first = get_np_pred_lin(np_pred, 48)
        print("PRED_48_DONE")
    elif q1 == "lin24":
        np_pred_first = get_np_pred_lin(np_pred, 24)
        print("PRED_24_DONE")
    elif q1 == "lin12":
        np_pred_first = get_np_pred_lin(np_pred, 12)
        print("PRED_12_DONE")
    
    if q2 == "lin96":
        np_pred_second = get_np_pred_lin(np_pred, 96)
        print("PRED_96_DONE")
    elif q2 == "lin48":
        np_pred_second = get_np_pred_lin(np_pred, 48)
        print("PRED_48_DONE")
    elif q2 == "lin24":
        np_pred_second = get_np_pred_lin(np_pred, 24)
        print("PRED_24_DONE")
    elif q2 == "lin12":
        np_pred_second = get_np_pred_lin(np_pred, 12)
        print("PRED_12_DONE")
    elif q2 == "none":
        np_pred_second = np.zeros(np_pred.shape)
        print("NO_DONE")
     
    
    print("COMBI", len(loss_points))
    
    # loss_points에서 수집한 도트들을 비교 -> 최소 MSE, 최소 MAE 검색, 최소 SMAE 검색
    loss_points_map = [] # a,b,c, mae, mse, smae, std_ratio, slope_ratio
    
    # loss_points에서 수집한 도트들을 비교 -> 최소 MSE, 최소 MAE 검색, 최소 SMAE 검색
    for j, (a,b) in enumerate(loss_points):
        res_temp = a*np_pred + b*np_pred_first + (1-a-b)*np_pred_second
        mse_step = MSE(res_temp, np_true)
        mae_step = MAE(res_temp, np_true)
        smae_step = SMAE(res_temp, np_true)
        # std_step = STD_RATIO(res_temp, np_true)
        # slope_step = SLOPE_RATIO(res_temp, np_true)
        
        # loss_points_map.append({"cnt": j, "a":a,"b":b,"MSE":mse_step,"MAE": mae_step,"SMAE": smae_step, "STD_RATIO": std_step, "slope_step": slope_step})
        loss_points_map.append({"cnt": j, "a":a,"b":b,"MSE":mse_step,"MAE": mae_step,"SMAE": smae_step})
        
    # MSE 기준으로 정렬
    main_key = "MSE" 
    new_loss_points_map = sorted(loss_points_map, key=lambda x: x[main_key])
    a, b = new_loss_points_map[0]["a"], new_loss_points_map[0]["b"]
    
    # 마지막으로 비교
    final_res = a*np_pred + b* np_pred_first + (1-a-b)*np_pred_second
    # final_res = a*np_pred + (1-a)*np_pred_lin
    
    # 메트릭 비교하기 (원본 iTransformer)
    with open(f'run_ensenble_txt_{setting_path}_combi_{q1}_{q2}.txt', 'w', encoding='utf8') as A:
        wr = "TRAIN_PRED\n"
        wr += f"{MSE(np_pred, np_true), MAE(np_pred, np_true), SMAE(np_pred, np_true), STD_RATIO(np_pred, np_true), SLOPE_RATIO(np_pred, np_true)} \n"
        wr += "TRAIN_ENSEMBLE_PRED\n"
        wr += f"{MSE(final_res, np_true), MAE(final_res, np_true), SMAE(final_res, np_true), STD_RATIO(final_res, np_true), SLOPE_RATIO(final_res, np_true)}\n"
        wr += "LIN_PRED\n"
        wr += "TRAIN_PRED_FIRST\n"
        wr += f"{MSE(np_pred_first, np_true), MAE(np_pred_first, np_true), SMAE(np_pred_first, np_true), STD_RATIO(np_pred_first, np_true), SLOPE_RATIO(np_pred_first, np_true)}\n"
        wr += "TRAIN_PRED_SECOND\n"
        wr += f"{MSE(np_pred_second, np_true), MAE(np_pred_second, np_true), SMAE(np_pred_second, np_true), STD_RATIO(np_pred_second, np_true), SLOPE_RATIO(np_pred_second, np_true)}\n"
        wr += f"loss_combi : {loss_points}\n"
        A.write(wr)
    
    # 메트릭 저장
    # metric_path = f"./results/{setting_path}"
    # metric_ensemble = [MSE(np_pred, np_true), MAE(np_pred, np_true), SMAE(np_pred, np_true), REC_CORR(np_pred, np_true), STD_RATIO(np_pred, np_true), SLOPE_RATIO(np_pred, np_true)]
    # np.save(metric_path + "metrics_ensemble.npy", metric_ensemble)
    # np.save(metric_path + "pred_ensemble.npy", final_res)
    # np.save(metric_path + "coef_col.npy", loss_points)
    # np.save(metric_path + "coef_metric.npy", loss_points_map)
    # np.save(metric_path + "pred_lin24_2.npy", np_pred_lin_24)
    
    print("WORK DONE")
    print()

print("FINISHED")


