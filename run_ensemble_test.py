import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import time
import json
import re

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
from utils.tools import EarlyStopping

from utils.metrics import *
from utils.tools import get_Xval, res_lin_reg, get_res_lin, zero_model, sigmoid_inverse, sigmoid
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

# 스크립트 단번에 호출하는 방법
scripts_texts = ""
script_path = "./scripts/script_ensemble_iTransformer_exchange.sh"

with open(script_path, 'r', encoding='utf8') as W:
    scripts_texts = W.read()

# 파라미터 치환
search_root_path = re.search('root_path=(.*)', scripts_texts).group(1)
search_data_path = re.search('data_path=(.*)', scripts_texts).group(1)
search_base_idea = re.search('base_idea=(.*)', scripts_texts).group(1)
search_model_name = re.search('model_name=(.*)', scripts_texts).group(1)

search_texts_part = {"root_path": search_root_path, "data_path": search_data_path, "base_idea": search_base_idea, "model_name":search_model_name}

script_boundary = re.finditer(r"#START_SCRIPT_\d\n(.*?)\n#", scripts_texts, re.DOTALL)

script_pairs = []
for sc_part in script_boundary:
    part_text = sc_part.group(1)
    part_text = part_text.replace('python -u run.py \\', '')
    for key, val in search_texts_part.items():
        part_text = part_text.replace(f"${{{key}}}", val)
    part_text = part_text[3:]
    part_text = part_text.replace('\\\n', '\n')
    part_text = part_text.replace('"', '')
    search_script_name = re.search('--checkpoint_name (.*) \s', part_text).group(1)
    # search_script_name = search_script_name.replace('"', '')
    script_pairs.append((part_text, search_script_name))

args_list = [] # argument list

# 스크립트 한번에 호출하기
for c in range(len(script_pairs)):
    arg = parser.parse_args(script_pairs[c][0].split())
    arg.use_gpu = True if torch.cuda.is_available() and arg.use_gpu else False
    
    if arg.use_gpu and arg.use_multi_gpu:
        arg.devices = arg.devices.replace(' ', '')
        device_ids = arg.devices.split(',')
        arg.device_ids = [int(id_) for id_ in device_ids]
        arg.gpu = arg.device_ids[0]
    
    args_list.append(arg)

scripts_new = """--task_name long_term_forecast \
  --is_training 0 \
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
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1"""

args_list2 = []
arg = parser.parse_args(scripts_new.split())
arg.use_gpu = True if torch.cuda.is_available() and arg.use_gpu else False

if arg.use_gpu and arg.use_multi_gpu:
    arg.devices = arg.devices.replace(' ', '')
    device_ids = arg.devices.split(',')
    arg.device_ids = [int(id_) for id_ in device_ids]
    arg.gpu = arg.device_ids[0]

args_list2.append(arg)


exchange_96_96_new = "long_term_forecast_COMPARE_iTransformer_linear_Exchange_96_96_Mod-iTransformer_linear_data-exchange_rate.csv_(96to96)_0(1729900000)"

setting_pairs = [
    (exchange_96_96_new, args_list2[0])
]

setting_pairs = [
    (script_pairs[r][1], args_list[r]) for r in range(len(script_pairs))
]


idx = 0 # 순서
col_count = 4 # 한 에포크당 수집 데이터 수
num_epochs = 3 # 에포크 ㅅ횟수
use_gpu = 0 # 사용 GPU 번호 - 오류 잡기 위해 
# tuple_test
q1, q2, q3, q4 = "lin96", "lin24", "none", "none"
a_init , b_init, c_init, d_init = sigmoid_inverse(0.05), sigmoid_inverse(0.05) , -100, -100  # 초기값(sigmoid로변환할  것 감안)  
lr = 0.1

q1, q2, q3, q4 = q1.lower(), q2.lower(), q3.lower(), q4.lower()

for idx in range(1):
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
    exp_model.model.load_state_dict(model, strict=True)
    exp_model.model.eval()

    dataset_input, dataset_input_loader = exp_model._get_data('train')
    dataset_input_val, dataset_input_val_loader = exp_model._get_data('val')
    dataset_input_test, dataset_input_test_loader = exp_model._get_data('test')
    
    
    X_old, X_new, X_concat = get_Xval(args.seq_len, args.pred_len)

    # exp_model.test(setting_path, test=1)

    # Combination 모델 제작, 2단계/3단계 대응
    
    class CombinedModel(nn.Module):
        # 모델 정의 - 
        def __init__(self, res_A, res_B, res_C, res_D):
            super(CombinedModel, self).__init__()
            self.res_A = res_A  # lin first
            self.res_B = res_B  # lin second
            self.res_C = res_C  # lin third
            self.res_D = res_D  # lin fourth

            self.a = nn.Parameter(torch.ones(1, device=device)*a_init, requires_grad=True)
            if res_B is not None:
                self.b = nn.Parameter(torch.ones(1, device=device)*b_init, requires_grad=True)
            else:
                self.b = nn.Parameter(torch.ones(1, device=device)*(-100))
            if res_C is not None:
                self.c = nn.Parameter(torch.ones(1, device=device)*c_init, requires_grad=True)
            else:
                self.c = nn.Parameter(torch.ones(1, device=device)*(-100))
            if res_D is not None:
                self.d = nn.Parameter(torch.ones(1, device=device)*d_init, requires_grad=True)
            else:
                self.d = nn.Parameter(torch.ones(1, device=device)*(-100))
            
        
        def set_a(self, val):
            # nn.Parameter를 다시 생성하지 않고 값을 설정
            with torch.no_grad():
                self.a.copy_(torch.tensor([val], device=device))
    
        def set_b(self, val):
            # nn.Parameter를 다시 생성하지 않고 값을 설정
            with torch.no_grad():
                self.b.copy_(torch.tensor([val], device=device))
        
        def set_c(self, val):
            # nn.Parameter를 다시 생성하지 않고 값을 설정
            with torch.no_grad():
                self.c.copy_(torch.tensor([val], device=device))
        
        def set_d(self, val):
            # nn.Parameter를 다시 생성하지 않고 값을 설정
            with torch.no_grad():
                self.d.copy_(torch.tensor([val], device=device))
                
        def forward(self, x, x_mark, y, y_mark):
            x_mark = x_mark.float().to(x.device)
            y = y.float().to(x.device)
            y_mark = y_mark.float().to(x.device)
            output_Z = exp_model.model(x, x_mark, y, y_mark).permute(0,2,1)
            # output_Z = res_iTransformer(x)
            output_A = self.res_A(x)
            
            # Apply sigmoid to ensure non-negative coefficients in range (0, 1)
            a_sigmoid = torch.sigmoid(self.a)

            z = 1 - a_sigmoid
            combined_output = z * output_Z + a_sigmoid * output_A

            if self.res_B is not None:
                output_B = self.res_B(x)
                b_sigmoid = torch.sigmoid(self.b)
                z = 1- a_sigmoid - b_sigmoid
                combined_output = z * output_Z + a_sigmoid * output_A + b_sigmoid* output_B

                if self.res_C is not None:
                    output_C = self.res_C(x)
                    c_sigmoid = torch.sigmoid(self.c)
                    z = 1- a_sigmoid - b_sigmoid - c_sigmoid
                    combined_output = z * output_Z + a_sigmoid * output_A + b_sigmoid* output_B + c_sigmoid * output_C

                    if self.res_D is not None:
                        output_D = self.res_D(x)
                        d_sigmoid = torch.sigmoid(self.d)
                        z = 1- a_sigmoid - b_sigmoid - c_sigmoid - d_sigmoid
                        combined_output = z * output_Z + a_sigmoid * output_A + b_sigmoid* output_B + c_sigmoid * output_C + d_sigmoid * output_D
        
            return combined_output
        
        def get_result(self):

            res = [torch.sigmoid(self.a).detach().cpu().numpy()[0]]

            if self.res_B is not None:
                res.append(torch.sigmoid(self.b).detach().cpu().numpy()[0])
            
            if self.res_C is not None:
                res.append(torch.sigmoid(self.c).detach().cpu().numpy()[0])
            
            if self.res_D is not None:
                res.append(torch.sigmoid(self.d).detach().cpu().numpy()[0])

            return tuple(res)
           
    
    def res_iTransformer(batch_x): # S 
        B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
        # decoder input
        global batch_x_train, batch_y_train

        batch_x_train_temp, batch_y_train_temp = batch_x_train[:B], batch_y_train[:B]

        dec_inp = torch.zeros(B, len(X_new), N).float().to(device)
        dec_inp = torch.cat([batch_x[:, -args.label_len:, :], dec_inp], dim=1).float().to(device)

        return exp_model.model(batch_x, batch_x_train_temp, dec_inp, batch_y_train_temp).permute(0, 2, 1) 
    
    
    QMAP_FN = {
      "none": None,
    }

    QMAP_FN_NUMS = [6, 12] + [k*24 for k in range(1, args.pred_len//24 + 1)]

    for num in QMAP_FN_NUMS:
        QMAP_FN[f"lin{num}"] = get_res_lin(num, args.pred_len)

         
    # 우선 train_set의 data_exchange를 바탕으로 측정값 참값 가져기
    # 트레인 데이터셋을 테스트해서 결과 받기, test 함수에서 가져옴

    start_time = time.time() # 시작타
    print(f"iteration start, time : {time.localtime()}")
    

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_loader):

        if i == 0:
            batch_x_train, batch_y_train = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
    
        if (i+1)%100==0:
            print(f"step {i+1} completed")
    
    input_len = i + 1
    
    # operation 끝
    data_col_time = time.time()
    print(f"경과 시간: {data_col_time-start_time}")

    # 실제 데이터 셋 호출
    result_list = ['pred.npy', 'true.npy']
    result_path = './results/'
    np_pred = np.load(f"{result_path}{setting_path}/{result_list[0]}")
    np_true = np.load(f"{result_path}{setting_path}/{result_list[1]}")

    print(f"test iteration start, time : {time.localtime()}")
    batch_x_new, batch_y_new = None, None

    batch_x_shapes = []

    # 직접 모델 실험 및 데이터 정리
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_test_loader):

        if i == 0:
            batch_x_test, batch_y_test = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
        
        with torch.no_grad():
        
            if (i+1)%100==0:
                print(f"step {i+1} completed")
    
    input_len_te = i + 1

    data_col_time = time.time()
    print(f"경과 시간: {data_col_time-start_time}")

    # 모델 실험
    combine_model_test = CombinedModel(QMAP_FN[q1], QMAP_FN[q2], QMAP_FN[q3], QMAP_FN[q4])

    # combine_model_test training
    combine_model_test.train()
    torch.nn.utils.clip_grad_norm_(combine_model_test.parameters(), max_norm=1) # gradient clipping - 크기 제한
    
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam([combine_model_test.a,combine_model_test.b], lr=lr, weight_decay=1e-3)
    optimizer = torch.optim.SGD([combine_model_test.a,combine_model_test.b], lr=lr, momentum=0.96)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 직접 메트릭 계산

    
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
                outputs = combine_model_test(batch_x, batch_x_mark, batch_y, batch_y_mark).permute(0,2,1)
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
    input_len_test_div = int(np.ceil(input_len_te / (col_count - 1)))
    
    print("INPUT_LEN", input_len, input_len_div)
    # early_stopping = EarlyStopping(patience=2, verbose=True)
    for epoch in range(num_epochs):
        cnt = 0
        train_loss = []
        # exp_model.train()

        
        path = os.path.join(args.checkpoints, setting_path)
        if not os.path.exists(path):
            os.makedirs(path)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_test_loader):
            cnt += 1
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            targets = batch_y[:, -args.pred_len:, :].to(device)
            optimizer.zero_grad()
            outputs = combine_model_test(batch_x, batch_x_mark, batch_y, batch_y_mark).permute(0,2,1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
            if (cnt+1) % 50 == 0:
                print(f"{cnt+1}th batch done, loss {loss}")
            if i == input_len_te -1 or i % input_len_test_div == 0:
                tup = combine_model_test.get_result()
                print(f"STEP {i}", combine_model_test.get_result(), f"loss {loss}" )
                loss_points.append(tup)
                train_loss_res = vali(dataset_input, dataset_input_test_loader, criterion)
                vali_loss = vali(dataset_input_test, dataset_input_test_loader, criterion)
                print("train_loss, vali_loss:", train_loss_res, vali_loss)
                # scheduler.step(vali_loss)

        print("="*50)
        print(f"Epoch {epoch+1} DONE")
        print()
        train_loss = [v.item() for v in train_loss]
        train_loss = np.average(train_loss)
        vali_loss = vali(dataset_input_test, dataset_input_test_loader, criterion)
        # print("vali_loss:", vali_loss)
        # print()
        print(tup)
        # early_stopping(vali_loss, combine_model_test, path)
        # if early_stopping.early_stop:
        #    print("Early stopping")
        #    break
        model_path = path + '/' + 'checkpoint_ensenble.pth'
    
    # 훈련 결과 도출
    combine_model_test.eval()
    
    # 계수 수집 끝
    end_time = time.time()
    print(f"Find Coefficients process done. Spent :{end_time - start_time}")

    

    def get_np_pred_lin(reg_size=96):
        B, L, N = np_pred.shape  # B: Batch size, L: Sequence length, N: Number of variables

        # Prepare the regression input for the last `reg_size` elements from X_old
        X_torch = torch.tensor(X_old[-reg_size:], dtype=torch.float32, device=device)  # Shape: [reg_size, 2]

        # Initialize tensor for storing results
        lin_result = torch.zeros(np_pred.shape, device=device)  # Shape: [B, pred_len, N]

        # Convert X_new to a tensor once outside the loop
        X_new_torch = torch.tensor(X_new, dtype=torch.float32, device=device)  # Shape: [pred_len, 2]

        # Perform batched linear regression for each batch and variable
        for idx in range(B):
            # Get the y_torch values for all variables in one operation
            y_torch = torch.Tensor(dataset_input_test[idx][0][-reg_size:, :]).to(device)

            # Compute least squares in batch for all variables at once
            # torch.linalg.lstsq can handle multiple right-hand sides (N variables)
            w = torch.linalg.lstsq(X_torch, y_torch).solution  # Shape: [2, N] (2 coefficients per variable)

            # Make predictions over the entire X_new using batch matrix multiplication
            lin_result[idx, :, :] = X_new_torch @ w  # Shape: [pred_len, N]

        return lin_result.detach().cpu().numpy()  
    
    
    # q1, q2가 잇을 때

    if re.match("lin(\d+)", q1):
        new_num_1 = int(re.match("lin(\d+)", q1).group(1))
        np_pred_first = get_np_pred_lin(new_num_1)
    else:
        new_num_1 = 0
        np_pred_second = np.zeros(np_pred.shape)
    
    if re.match("lin(\d+)", q2):
        new_num_2 = int(re.match("lin(\d+)", q2).group(1))
        np_pred_second = get_np_pred_lin(new_num_2)
    else:
        new_num_2 = 0
        np_pred_second = np.zeros(np_pred.shape)

    if re.match("lin(\d+)", q3):
        new_num_3 = int(re.match("lin(\d+)", q3).group(1))
        np_pred_third = get_np_pred_lin(new_num_3)
    else:
        new_num_3 = 0
        np_pred_third = np.zeros(np_pred.shape)

    if re.match("lin(\d+)", q4):
        new_num_4 = int(re.match("lin(\d+)", q3).group(1))
        np_pred_fourth = get_np_pred_lin(new_num_4)
    else:
        new_num_4 = 0
        np_pred_fourth = np.zeros(np_pred.shape)

    
    print("COMBI", len(loss_points))
    
    loss_points_map = [] # a,b,c, mae, mse, smae, std_ratio, slope_ratio

    np_pred_list = [np_pred_first, np_pred_second, np_pred_third, np_pred_fourth]
    
    # loss_points에서 수집한 도트들을 비교 -> 최소 MSE, 최소 MAE 검색, 최소 SMAE 검색
    print(np_pred.shape, np_pred_first.shape, np_pred_second.shape)
    for j, tup in enumerate(loss_points):
        tup_sum = sum(tup)
        res_temp = (1-tup_sum)*np_pred
        for r in range(len(tup)):
            res_temp = res_temp + tup[r]*np_pred_list[r]

        mse_step = MSE(res_temp, np_true)
        mae_step = MAE(res_temp, np_true)
        smae_step = SMAE(res_temp, np_true)

        coeff_tup = tup
        
        loss_points_map.append({"cnt": j, "tup": tup,"MSE":mse_step,"MAE": mae_step,"SMAE": smae_step, "res_temp": res_temp})
    
    # MSE 기준으로 정렬
    main_key = "MSE" 
    new_loss_points_map = sorted(loss_points_map, key=lambda x: x[main_key])
            
    # 마지막으로 비교
    final_res = new_loss_points_map[0]["res_temp"]
    tup = new_loss_points_map[0]["tup"]
    
    # 메트릭 비교하기 (원본 iTransformer)
    with open(f'run_ensenble_txt_{setting_path}_{q1}_{q2}_{time.time()}.txt', 'w', encoding='utf8') as A:
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
        wr += f"COEFFICIENTS : {1-sum(tup)}, {', '.join([str(tupval) for tupval in tup])}\n"
        A.write(wr)
    
    # 메트릭 저장
    metric_path = f"./results/{setting_path}/"
    metric_ensemble = [MSE(np_pred, np_true), MAE(np_pred, np_true), SMAE(np_pred, np_true), REC_CORR(np_pred, np_true), STD_RATIO(np_pred, np_true), SLOPE_RATIO(np_pred, np_true)]
    np.save(metric_path + "metrics_ensemble_te.npy", metric_ensemble)
    np.save(metric_path + "pred_ensemble_te.npy", final_res)
    np.save(metric_path + "coef_col_te.npy", loss_points)

    # loss_points_map 저장 전 res_temp 키 지우기
    for r in range(len(loss_points_map)):
        del loss_points_map[r]["res_temp"]
    
    np.save(metric_path + "coef_metric_te.npy", loss_points_map)

    # 그래픽 표현
    graphic_path = f"./test_results/{setting_path}/"
    x_old = np.array(range(-args.seq_len, 0))
    x_new = np.array(range(0, args.pred_len))
    x_concat = np.concatenate([x_old, x_new]) # 입력길이
    # 20단위로
    for idx in range(0, len(np_pred), 20):
        input_val = dataset_input_test[idx][0][:, -1] # 입력
        pred_val = np_pred[idx, :, -1] # 예측값
        true_val = np_true[idx, :, -1] # 실제값
        lin_val = np_pred_first[idx, :, -1] # 선형1
        lin_val2 = np_pred_second[idx, : ,-1] # 선형2
        lin_val_slope, lin_val_intercept = lin_val[1] - lin_val[0] , lin_val[0]
        lin_val2_slope, lin_val2_intercept = lin_val2[1] - lin_val2[0] , lin_val2[0]
        input_val_lin = np.array([t*lin_val_slope + lin_val_intercept for t in x_old])
        input_val_lin2 = np.array([t*lin_val2_slope + lin_val2_intercept for t in x_old])
        final_val = final_res[idx, :, -1] # 최종값
        pred_val = np.concatenate([input_val, pred_val])
        true_val = np.concatenate([input_val, true_val])
        lin_val = np.concatenate([input_val_lin, lin_val])
        lin_val2 = np.concatenate([input_val_lin2, lin_val2])
        final_val = np.concatenate([input_val, final_val])
        file_nm = f"combi_{idx}_test.pdf"
        plt.figure(figsize=(8,7))
        plt.plot(x_concat, true_val, 'g-', label="GroundTruth", linewidth=2)
        plt.plot(x_concat, pred_val, 'b-', label="Prediction_Basic", linewidth=2)
        plt.plot(x_concat, lin_val, 'y-', label="First Lin", linewidth=1)
        plt.plot(x_concat, lin_val2, 'y-', label="Second Lin", linewidth=1)
        plt.plot(x_concat, final_val, 'k-', label="Prediction_Final", linewidth=1)
        plt.legend()
        plt.savefig(f"{graphic_path}{file_nm}", bbox_inches="tight")

    
    print("WORK DONE")
    print()
    

print("FINISHED")


