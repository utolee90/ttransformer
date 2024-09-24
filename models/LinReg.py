import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp



class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decomposition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        # self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_coef = nn.Linear(configs.enc_in*2, configs.enc_in*2) # 상수 계수 추적.

        # linear regresion
        self.linear_regression = nn.Linear(1, 1, bias=True)

        # self.Linear_Seasonal.weight = nn.Parameter(
        #     (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)


    def encoder(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        B, N, L = trend_init.shape  # L은 시퀀스 길이(seq_len)

        # trend_output 계산
        trend_output = self.Linear_Trend(trend_init)

        # 시간 변수 생성
        X = np.array([[t] for t in range(-self.seq_len, 0)])  # X는 입력 feature, shape: [seq_len, 1]
        X_new = np.array([[t] for t in range(self.seq_len)])  # 예측을 위한 새로운 시간 변수
        
        # 각 배치와 변수에 대해 선형 회귀 해를 계산
        vals = [[self.linear_regression_direct(X, trend_output[idx, var , :]) for var in range(N)] for idx in range(B)]
        lin_result = [[self.linear_predict(X_new, vals[idx][var]) for var in range(N)] for idx in range(B)]

        # 결과를 3D 텐서로 변환
        lin_result = torch.stack([torch.stack(lin_result[idx], dim=0) for idx in range(B)], dim=0).to(self.device)

        # 최종 출력
        x = lin_result
        return x.permute(0, 2, 1)

    def linear_regression_direct(self, X, y):
        # 선형 회귀를 행렬 곱으로 직접 계산
        X_b = torch.cat([torch.ones((X.shape[0], 1)), torch.tensor(X, dtype=torch.float32)], dim=1).to(self.device)
        y_torch = torch.tensor(y, dtype=torch.float32).to(self.device).unsqueeze(1)
        XTX = X_b.T @ X_b
        XTy = X_b.T @ y_torch
        theta_best = torch.linalg.inv(XTX) @ XTy
        return theta_best
    
    def linear_regression_direct(self, X, y):
        """
        선형 회귀 해를 행렬 곱셈으로 직접 계산합니다.
        X: 입력 feature 행렬, shape: [seq_len, 1] 
        y: 타겟 값, shape: [seq_len]
        """
        # X에 bias term 추가 (ones column 추가)
        X_b = torch.cat([torch.ones((X.shape[0], 1)), torch.tensor(X, dtype=torch.float32, requires_grad=True)], dim=1).to(self.device)
        
        # y를 텐서로 변환
        y_torch = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(self.device).unsqueeze(1)

        # (X^T X) theta = X^T y 를 풀기 위해 solve() 사용
        XTX = X_b.T @ X_b  # X^T X
        XTy = X_b.T @ y_torch  # X^T y

        # torch.linalg.solve()을 사용하여 해 구하기
        theta_best = torch.linalg.solve(XTX, XTy)

        # θ (bias, weight) 반환
        return theta_best


    def linear_predict(self, X_new, theta_best):
        # 학습된 θ로 예측 수행
        X_new_b = torch.cat([torch.ones((X_new.shape[0], 1)), torch.tensor(X_new, dtype=torch.float32)], dim=1).to(self.device)
        y_pred = X_new_b @ theta_best
        return y_pred.squeeze(1)


    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    """
    def linear_regression_lstsq(self, X, y):
        # X: input features (shape: [seq_len, 1])
        # y: target values (shape: [seq_len])
        
        # X_b는 X에 bias term을 추가한 텐서 (ones column 추가)
        X_b = torch.cat([torch.ones((X.shape[0], 1)), torch.tensor(X, dtype=torch.float32)], dim=1).to(self.device)
        
        # y를 텐서로 변환 (PyTorch 사용)
        y_torch = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # 선형 회귀 계수 구하기 (최소제곱법을 통한 해)
        res = torch.linalg.lstsq(y_torch.unsqueeze(1), X_b)

        theta_best = res.solution
        
        # theta_best는 [bias, weight] 형태로 반환됨
        return theta_best

    def linear_predict(self, X_new, theta_best):
        # X_new: 새로운 input features (shape: [pred_len, 1])
        # theta_best: learned linear regression parameters (shape: [2, 1])

        # X_new에 bias term 추가
        X_new_b = torch.cat([torch.ones((X_new.shape[0], 1)), torch.tensor(X_new, dtype=torch.float32)], dim=1).to(self.device)
        # print(theta_best.transpose(0,1))
        # 예측값 계산
        y_pred = X_new_b.mm(theta_best.transpose(0,1))  # X_new_b와 theta_best의 행렬 곱
        return y_pred.squeeze(1)  # 결과를 반환 (예측값 벡터)
    """

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
