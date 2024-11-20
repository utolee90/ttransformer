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

FIBONACCI_NUMS = [0,1,2,3,5,8,13,21,34,55,89,144,233,377,610] # 피보나치 수열

# fix random seed
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 스크립트 단번에 호출하는 방법
scripts_texts = ""
script_path = "./scripts/script_ensemble_DLinear_exchange_noshuffle.sh"

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


exchange_96_96_new = "long_term_forecast_COMPARE_iTransformer_linear_Exchange_96_96_Mod-iTransformer_linear_data-exchange_rate.csv_(96to96)_0(1729900000)"

setting_pairs = [
    (script_pairs[r][1], args_list[r]) for r in range(len(script_pairs))
]


col_count = 4 # 한 에포크당 수집 데이터 수
num_epochs = 3 # 에포크 ㅅ횟수
use_gpu = 0 # 사용 GPU 번호 - 오류 잡기 위해 
# tuple_test
q1, q2, q3, q4 = "lin96", "lin24", "none", "none"
a_init , b_init, c_init, d_init = sigmoid_inverse(0.05), sigmoid_inverse(0.05) , -100, -100  # 초기값(sigmoid로변환할  것 감안)  
lr = 0.005 #SGD 사용시에는 lr값을 충분히 키워서 쓸 것. Adam일 때는 0.01 정도가 적합
lr = 0.05

dropout_rate = 0.1 # 드롭아웃 비율

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

    # fibonacci 눈금 놓기 - 큰수부터 작은수 순서로
    fibonacci_grid = [args.seq_len - 1 - t for t in FIBONACCI_NUMS if args.seq_len - 1 - t >= 0]

    # 6가지 데이터 수집 - 순서 오차가 있을 수 있으니 확인할 것.
    np_train_input = [] # 입력값
    np_train_pred = [] # 모델 예측값
    np_train_true = [] # 실제 데이터값
    np_train_diff = [] # 오차
    np_train_diff_approx = [] # 오차 - 근사
    np_train_diff_coeff = [] # 오차 - 계수
    np_train_approx_sum = [] # 근사값 + 예측값
    np_train_combi_fibonacci = [] # 피보나치 수열으로 근사하기
    np_train_combi_fibonacci_approx = [] # 피보나치 수열로 근사한 계수 조합.
    np_train_combi_fibonacci_approx_i = [] # 피보나치 수열로 근사한 계수 조합 - 상수

    # 데이터 수집하기
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_loader):

        # batch_x, batch_y, 
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        B,L,N = batch_x.shape

        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        # encoder - decoder
    
        # use_amp도 사용하지 않음, 
        outputs = exp_model.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)

        initial = batch_x # input
        pred = outputs # model result
        true = batch_y # actual result
        diff = batch_y - pred # differenced
        
        # Prepare the regression input for the last `reg_size` elements from X_old
        X_torch = torch.tensor(X_new, dtype=torch.float32, device=device)  # Shape: [reg_size, 2]
        # Initialize tensor for storing results
        diff_approx = torch.zeros((B, X_new.shape[0], N), device=device)  # Shape: [B, pred_len, N]
        
        w_stack = []
        for idx in range(B):
            y_torch = diff[idx, -args.pred_len:, :].to(device)

            w = torch.linalg.lstsq(X_torch, y_torch).solution
            w_stack.append(w.permute(0,1))
            diff_approx[idx, :, :] = X_torch @ w

        diff_coeff = torch.stack(w_stack, dim=0)

        approx_sum = pred + diff_approx

        np_train_input.append(initial)
        np_train_pred.append(pred)
        np_train_true.append(true)
        np_train_diff.append(diff)
        np_train_diff_approx.append(diff_approx)
        np_train_diff_coeff.append(diff_coeff)
        np_train_approx_sum.append(approx_sum)

        # 피보나치 그리드 값으로 도출하기
        initical_fibonacci = torch.concatenate([torch.ones([B,1,N]), initial[:, fibonacci_grid, :]], axis=1)
        np_train_combi_fibonacci.append(initical_fibonacci)

        if i == 0:
            batch_x_train, batch_y_train = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
    
        if (i+1)%100==0:
            print(f"step {i+1} completed")
    
    input_len = i + 1
    
    # operation 끝
    data_col_time = time.time()
    print(f"경과 시간: {data_col_time-start_time}")

    np_train_input = torch.concat(np_train_input, axis=0).detach().cpu().numpy()
    np_train_pred = torch.concat(np_train_pred, axis=0).detach().cpu().numpy()
    np_train_true = torch.concat(np_train_true, axis=0).detach().cpu().numpy()
    np_train_diff = torch.concat(np_train_diff, axis=0).detach().cpu().numpy()
    np_train_diff_approx = torch.concat(np_train_diff_approx, axis=0).detach().cpu().numpy()
    np_train_diff_coeff = torch.concat(np_train_diff_coeff, axis=0).detach().cpu().numpy()
    np_train_approx_sum = torch.concat(np_train_approx_sum, axis=0).detach().cpu().numpy()

    # 6가지 데이터 수집 - 순서 오차가 있을 수 있으니 확인할 것.
    np_val_input = [] # 입력값
    np_val_pred = [] # 모델 예측값
    np_val_true = [] # 실제 데이터값
    np_val_diff = [] # 오차
    np_val_diff_approx = [] # 오차 - 근사
    np_val_diff_coeff = [] # 오차 - 계수
    np_val_approx_sum = [] # 근사값 + 예측값

    # 직접 모델 실험 및 데이터 정리 - valid
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_val_loader):
        # batch_x, batch_y, 
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        B,L,N = batch_x.shape

        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        # encoder - decoder
    
        # use_amp도 사용하지 않음, 
        outputs = exp_model.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)

        initial = batch_x # input
        pred = outputs # model result
        true = batch_y # actual result
        diff = batch_y - pred # differenced
        
        # Prepare the regression input for the last `reg_size` elements from X_old
        X_torch = torch.tensor(X_new, dtype=torch.float32, device=device)  # Shape: [reg_size, 2]
        # Initialize tensor for storing results
        diff_approx = torch.zeros((B, X_new.shape[0], N), device=device)  # Shape: [B, pred_len, N]
        
        w_stack = []
        for idx in range(B):
            y_torch = diff[idx, -args.pred_len:, :].to(device)

            w = torch.linalg.lstsq(X_torch, y_torch).solution
            w_stack.append(w.permute(0,1))
            diff_approx[idx, :, :] = X_torch @ w

        diff_coeff = torch.stack(w_stack, dim=0)

        approx_sum = pred + diff_approx

        np_val_input.append(initial)
        np_val_pred.append(pred)
        np_val_true.append(true)
        np_val_diff.append(diff)
        np_val_diff_approx.append(diff_approx)
        np_val_diff_coeff.append(diff_coeff)
        np_val_approx_sum.append(approx_sum)

        if i == 0:
            batch_x_test, batch_y_test = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
        
        with torch.no_grad():
        
            if (i+1)%100==0:
                print(f"step {i+1} completed")
    
    np_val_input = torch.concat(np_val_input, axis=0).detach().cpu().numpy()
    np_val_pred = torch.concat(np_val_pred, axis=0).detach().cpu().numpy()
    np_val_true = torch.concat(np_val_true, axis=0).detach().cpu().numpy()
    np_val_diff = torch.concat(np_val_diff, axis=0).detach().cpu().numpy()
    np_val_diff_approx = torch.concat(np_val_diff_approx, axis=0).detach().cpu().numpy()
    np_val_diff_coeff = torch.concat(np_val_diff_coeff, axis=0).detach().cpu().numpy()
    np_val_approx_sum = torch.concat(np_val_approx_sum, axis=0).detach().cpu().numpy()
    
    input_len_val = i + 1

    # 6가지 데이터 수집 - 순서 오차가 있을 수 있으니 확인할 것.
    np_test_input = [] # 입력값
    np_test_pred = [] # 모델 예측값
    np_test_true = [] # 실제 데이터값
    np_test_diff = [] # 오차
    np_test_diff_approx = [] # 오차 - 근사
    np_test_diff_coeff = [] # 오차 - 계수
    np_test_approx_sum = [] # 근사값 + 예측값

    # 직접 모델 실험 및 데이터 정리
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_test_loader):

        # batch_x, batch_y, 
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        B,L,N = batch_x.shape

        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        # encoder - decoder
    
        # use_amp도 사용하지 않음, 
        outputs = exp_model.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)

        initial = batch_x # input
        pred = outputs # model result
        true = batch_y # actual result
        diff = batch_y - pred # differenced
        
        # Prepare the regression input for the last `reg_size` elements from X_old
        X_torch = torch.tensor(X_new, dtype=torch.float32, device=device)  # Shape: [reg_size, 2]
        # Initialize tensor for storing results
        diff_approx = torch.zeros((B, X_new.shape[0], N), device=device)  # Shape: [B, pred_len, N]

        
        w_stack = []
        for idx in range(B):
            y_torch = diff[idx, -args.pred_len:, :].to(device)

            w = torch.linalg.lstsq(X_torch, y_torch).solution
            w_stack.append(w.permute(0,1))
            diff_approx[idx, :, :] = X_torch @ w

        diff_coeff = torch.stack(w_stack, dim=0)
        approx_sum = diff_approx + pred

        np_test_input.append(initial)
        np_test_pred.append(pred)
        np_test_true.append(true)
        np_test_diff.append(diff)
        np_test_diff_approx.append(diff_approx)
        np_test_diff_coeff.append(diff_coeff)
        np_test_approx_sum.append(approx_sum)

        if i == 0:
            batch_x_test, batch_y_test = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
        
        with torch.no_grad():
        
            if (i+1)%100==0:
                print(f"step {i+1} completed")
    
    input_len_te = i + 1

    np_test_input = torch.concat(np_test_input, axis=0).detach().cpu().numpy()
    np_test_pred = torch.concat(np_test_pred, axis=0).detach().cpu().numpy()
    np_test_true = torch.concat(np_test_true, axis=0).detach().cpu().numpy()
    np_test_diff = torch.concat(np_test_diff, axis=0).detach().cpu().numpy()
    np_test_diff_approx = torch.concat(np_test_diff_approx, axis=0).detach().cpu().numpy()
    np_test_diff_coeff = torch.concat(np_test_diff_coeff, axis=0).detach().cpu().numpy()
    np_test_approx_sum = torch.concat(np_test_approx_sum, axis=0).detach().cpu().numpy()

    data_col_time = time.time()
    print(f"경과 시간: {data_col_time-start_time}")
    
    # 오브젝트 저장
    if not os.path.exists(f'./coef_results/{setting_path}'):
        os.makedirs(f'./coef_results/{setting_path}', exist_ok=True)
    
    save_path = f'./coef_results/{setting_path}/'

    # 오브젝트 저장 - train
    np.save(save_path + 'train_input.npy', np_train_input )
    np.save(save_path + 'train_pred.npy', np_train_pred )
    np.save(save_path + 'train_true.npy', np_train_true )
    np.save(save_path + 'train_diff.npy', np_train_diff )
    np.save(save_path + 'train_diff_approx.npy', np_train_diff_approx )
    np.save(save_path + 'train_diff_coeff.npy', np_train_diff_coeff )
    np.save(save_path + 'train_approx_sum.npy', np_train_approx_sum )

    #오브젝ㅌ 저장 - val
    np.save(save_path + 'val_input.npy', np_val_input )
    np.save(save_path + 'val_pred.npy', np_val_pred )
    np.save(save_path + 'val_true.npy', np_val_true )
    np.save(save_path + 'val_diff.npy', np_val_diff )
    np.save(save_path + 'val_diff_approx.npy', np_val_diff_approx )
    np.save(save_path + 'val_diff_coeff.npy', np_val_diff_coeff )
    np.save(save_path + 'val_approx_sum.npy', np_val_approx_sum )

    np.save(save_path + 'test_input.npy', np_test_input )
    np.save(save_path + 'test_pred.npy', np_test_pred )
    np.save(save_path + 'test_true.npy', np_test_true )
    np.save(save_path + 'test_diff.npy', np_test_diff )
    np.save(save_path + 'test_diff_approx.npy', np_test_diff_approx )
    np.save(save_path + 'test_diff_coeff.npy', np_test_diff_coeff )
    np.save(save_path + 'test_approx_sum.npy', np_test_approx_sum )

    # epoch별로 테스트하기
    earlystopping = EarlyStopping(patience=5, path=save_path + 'checkpoint.pth')

    


    # vali 함수
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


    for epoch in range(num_epochs):

        # train 데이터에서 a + tb 계수 추출하기
        # 데이터 수집하기
        for j, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataset_input_loader):

            # 1- dropout_rate만큼만 데이터 수집
            if random.random() > 1 - dropout_rate:
                continue

            # batch_x, batch_y, 
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            B,L,N = batch_x.shape

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
        
            # use_amp도 사용하지 않음, 
            outputs = exp_model.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            initial = batch_x # input
            pred = outputs # model result
            true = batch_y # actual result
            diff = batch_y - pred # differenced
            
            # Prepare the regression input for the last `reg_size` elements from X_old
            X_torch = torch.tensor(X_new, dtype=torch.float32, device=device)  # Shape: [reg_size, 2]
            # Initialize tensor for storing results
            diff_approx = torch.zeros((B, X_new.shape[0], N), device=device)  # Shape: [B, pred_len, N]
            
            w_stack = []
            for idx in range(B):
                y_torch = diff[idx, -args.pred_len:, :].to(device)

                w = torch.linalg.lstsq(X_torch, y_torch).solution
                w_stack.append(w.permute(0,1))
                diff_approx[idx, :, :] = X_torch @ w

            diff_coeff = torch.stack(w_stack, dim=0)

            approx_sum = pred + diff_approx

            np_train_input.append(initial)
            np_train_pred.append(pred)
            np_train_true.append(true)
            np_train_diff.append(diff)
            np_train_diff_approx.append(diff_approx)
            np_train_diff_coeff.append(diff_coeff)
            np_train_approx_sum.append(approx_sum)

            # 피보나치 그리드 값으로 도출하기
            initical_fibonacci = torch.concatenate([torch.ones([B,1,N]), initial[:, fibonacci_grid, :]], axis=1)
            np_train_combi_fibonacci.append(initical_fibonacci)

            if j == 0:
                batch_x_train, batch_y_train = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
        
            if (j+1)%100==0:
                print(f"step {j+1} completed")
        


print("FINISHED")


