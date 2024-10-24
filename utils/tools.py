import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
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

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
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


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

# 선형 회귀 계산
def linear_regression_direct(X, y, device=None):
    """
    선형 회귀 해를 행렬 곱셈으로 직접 계산합니다.
    X: 입력 feature 행렬, shape: [seq_len, 1] 
    y: 타겟 값, shape: [seq_len]
    """
    # Tensor

    # X가 numpy array인 경우 텐서로 변환
    # if isinstance(X, np.ndarray):
    #    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    # if isinstance(y, np.ndarray):
    #    y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
 
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


# xval 도출
def get_Xval(seq_len, pred_len):
    X_old = np.array([[t, 1] for t in range(-seq_len, 0)])  # X_old는 입력 feature, shape: [seq_len, 1]
    X_new = np.array([[t, 1] for t in range(pred_len)])  # 예측을 위한 새로운 시간 변수
    # print(X_old.shape, X_new.shape)
    X_concat = np.concatenate([X_old, X_new], axis=0).reshape(-1)
    return X_old, X_new, X_concat

# batch_x -> lin_reg 도출
def res_lin_reg(batch_x, seq_len, pred_len):

    B, L, N = batch_x.shape  # B: Batch size, L: Sequence length, N: Number of variables

    device = batch_x.device

    X_old, X_new, X_concat = get_Xval(seq_len, pred_len)


    # Prepare the regression input for the last `reg_size` elements from X_old
    X_torch = torch.tensor(X_old, dtype=torch.float32, device=device)  # Shape: [reg_size, 2]

    # Initialize tensor for storing results
    lin_result = torch.zeros((B, X_new.shape[0], N), device=device)  # Shape: [B, pred_len, N]

    # Convert X_new to a tensor once outside the loop
    X_new_torch = torch.tensor(X_new, dtype=torch.float32, device=device)  # Shape: [pred_len, 2]

    # Perform batched linear regression for each batch and variable
    for idx in range(B):
        # Get the y_torch values for all variables in one operation
        y_torch = batch_x[idx, -seq_len:, :].to(device)  # Shape: [reg_size, N]

        # Compute least squares in batch for all variables at once
        # torch.linalg.lstsq can handle multiple right-hand sides (N variables)
        w = torch.linalg.lstsq(X_torch, y_torch).solution  # Shape: [2, N] (2 coefficients per variable)

        # Make predictions over the entire X_new using batch matrix multiplication
        lin_result[idx, :, :] = X_new_torch @ w  # Shape: [pred_len, N]

    return lin_result.permute(0, 2, 1)  # Return in shape [B, N, pred_len]

# res_len 함수 추가
def get_res_lin(seq_len, pred_len):

    def fn(batch_x):
        return res_lin_reg(batch_x, seq_len, pred_len)
    
    return fn

# 0함수 도출
def zero_model(batch_x, pred_len): # S 길이
    B, L, N = batch_x.shape  # L은 시퀀스 길이(seq_len)
    return torch.zeros(B, pred_len, N)

# 시그모이드 역함수
def sigmoid_inverse(y):
    # y는 0과 1 사이의 값이어야 합니다.
    return np.log(y / (1 - y))

# 시그모이드 함수
sigmoid = lambda x: 1 / (1 + np.exp(-x)) # exp는 지수 함수를 의미함