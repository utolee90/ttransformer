import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler


# 선형 회귀 계산
def linear_regression_direct(X, y, device=None):
    """
    선형 회귀 해를 행렬 곱셈으로 직접 계산합니다.
    X: 입력 feature 행렬, shape: [seq_len, 1] 
    y: 타겟 값, shape: [seq_len]
    """
 
    # X에 bias term 추가 (ones column 추가)
    X_b = torch.cat([torch.ones((X.shape[0], 1)), torch.tensor(X, dtype=torch.float32, requires_grad=True)], dim=1).to(device)
    
    # y를 텐서로 변환
    y_torch = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device).unsqueeze(1)

    # (X^T X) theta = X^T y 를 풀기 위해 solve() 사용
    XTX = X_b.T @ X_b  # X^T X
    XTy = X_b.T @ y_torch  # X^T y

    # torch.linalg.solve()을 사용하여 해 구하기
    theta_best = torch.linalg.solve(XTX, XTy)

    # θ (bias, weight) 반환
    return theta_best

def linear_predict(X_new, theta_best, device=None):
    """
    새로운 입력 데이터 X_new에 대해 예측값을 계산합니다.
    X_new: 새로운 input features, shape: [pred_len, 1]
    theta_best: 학습된 선형 회귀 계수 (bias와 weight)
    """
    # X_new에 bias term 추가

    # X가 numpy array인 경우 텐서로 변환
    # if isinstance(X_new, np.ndarray):
    #     X_new = torch.tensor(X_new, dtype=torch.float32, requires_grad=True).to(device)
    # if isinstance(theta_best, np.ndarray):
    #    theta_best = torch.tensor(theta_best, dtype=torch.float32, requires_grad=True).to(device)

    X_new_b = torch.cat([torch.ones((X_new.shape[0], 1)), torch.tensor(X_new, dtype=torch.float32)], dim=1).to(device)

    # θ (bias, weight)를 사용하여 예측값 계산
    y_pred = X_new_b @ theta_best  # 행렬 곱셈
    return y_pred.squeeze(1)  # 예측 결과를 반환


setting_path = "long_term_forecast_iTransformer_weather_96_96_Mod-iTransformer_data-weather.csv_(96to96)_0(1727354116)"

q1, q2 = "lin96", "lin24"

# 실제 데이터 셋 호출
result_list = ['pred.npy', 'true.npy']
result_path = './results/'
np_pred = np.load(f"{result_path}{setting_path}/{result_list[0]}")
np_true = np.load(f"{result_path}{setting_path}/{result_list[1]}")

X = np.array([[t] for t in range(-96, 0)])  # X는 입력 feature, shape: [seq_len, 1]
X_new = np.array([[t] for t in range(len(np_pred[0]))])  # 예측을 위한 새로운 시간 변수
X_concat = np.concatenate([X, X_new], axis=0).reshape(-1)

scaler = StandardScaler()
df_raw = pd.read_csv('./dataset/weather/weather.csv')
cols = list(df_raw.columns)
cols.remove('date')
cols.remove('OT')
num_total = len(df_raw)
num_train = int(num_total * 0.7)
num_test = int(num_total * 0.2)
num_vali = num_total - num_train - num_test
border1s = [0, num_train - 96, num_total - num_test - 96]
border2s = [num_train, num_train + num_vali, num_total]
border1 = border1s[2]
border2 = border2s[2]
cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]

train_data = df_data[border1s[0]:border2s[0]]
scaler.fit(train_data.values)
data = scaler.transform(df_data.values)
data_x = data[border1:border2]

dataset_input_test = [data_x[j:j+96, :] for j in range(10635-96+1)]

use_gpu = 3
device = torch.device(f"cuda:{use_gpu}")

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


