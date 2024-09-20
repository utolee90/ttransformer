import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.Autoformer_EncDec import series_decomp
from sklearn.linear_model import LinearRegression
from utils.metrics import SMAE, MSE
from scipy.optimize import minimize



class LinearApproximation(nn.Module):
    def __init__(self, num_var):
        super(LinearApproximation, self).__init__()
        self.linear = nn.Linear(1, num_var, bias=True)
    
    def forward(self, t):
        return self.linear(t)
    


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        
        # Series decomposition block from Autoformer
        self.decomposition = series_decomp(configs.moving_avg)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Embedding
        self.enc_embedding_s = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # linear_adjustment - 각 변수마다 x_i = x_i(0) + x_i(1)t로 보정
        self.linear_adjustment = LinearApproximation(configs.enc_in)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Encoder_seasonal
        self.encoder_s = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            self.projection_s = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            self.bias_adjustment = nn.Linear(configs.seq_len, configs.seq_len, bias=True) # 오차보정
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    """
    def train_linear_approximation(self, x_enc, num_epochs=1000, lr=0.001):
        B, T, N = x_enc.shape

        # time_step
        t = torch.arange(self.seq_len) - (self.seq_len -1) * torch.ones(self.seq_len) # (-95,..., 0)
        t = t.float().unsqueeze(0).unsqueeze(-1) # (1, time, 1)
        t = t.repeat(B, 1, 1) # BATCH_SIZE, TIME_STEPs, 1

        optimizer = torch.optim.SGD(self.linear_approximation.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epochs in range(num_epochs):
            linear_approx_output = self.linear_approximation(t)
            loss = criterion(linear_approx_output, x_enc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        a1 = self.linear_approximation.linear.weight.data
        a0 = self.linear_approximation.linear.bias.data

        return a0, a1
    """

    # 선형회귀 계수 구하기
    def linear_regression_lstsq(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.lstsq(X_b, y, rcond=None)[0]
        return theta_best

    # 선형회귀 예측
    def linear_predict(self, X_new, theta_best):
        X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new_b.dot(theta_best)
    
    # 손실함수
    def loss_function(self, weights, y_true, y_pred1, y_pred2):
        return MSE(weights[0] * y_pred1 + weights[1] * y_pred2, y_true)
    
    def optimize_weights(self, y_true, y_pred1, y_pred2):
        # 초기 가중치 (0.5, 0.5)로 설정
        initial_weights = [0.5, 0.5]
        # 가중치의 합이 1이 되도록 제약 조건 설정
        constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}
        # 가중치는 양수로 제한
        bounds = [(0, 1), (0, 1)]
        
        result = minimize(self.loss_function, initial_weights, args=(y_true, y_pred1, y_pred2), 
                        bounds=bounds, constraints=constraints)
        return result.x


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        B, T, N = x_enc.shape

        # Decomposition
        s_enc, t_enc = self.decomposition(x_enc)
        t_enc_copy = t_enc.clone().detach().cpu().numpy() # t_enc -> 

        # a0, a1 = self.train_linear_approximation(x_enc)

        # time_step_out
        # t_out = torch.arange(self.pred_len)  # (-95,..., 0)
        # t_out = t_out.float().unsqueeze(0).unsqueeze(-1) # (1, time, 1)
        # t_out = t_out.repeat(B, 1, N) # BATCH_SIZE, TIME_STEPs, Variable


        # Normalization from Non-stationary Transformer - s, t
        s_means = s_enc.mean(1, keepdim=True).detach()
        s_enc = s_enc - s_means
        s_stdev = torch.sqrt(torch.var(s_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        s_enc /= s_stdev

        t_means = t_enc.mean(1, keepdim=True).detach()
        t_enc = t_enc - t_means
        t_stdev = torch.sqrt(torch.var(t_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        t_enc /= t_stdev


        # Embedding
        s_enc_out = self.enc_embedding_s(s_enc, x_mark_enc)
        s_enc_out, s_attns = self.encoder_s(s_enc_out, attn_mask=None)
        t_enc_out = self.enc_embedding(t_enc, x_mark_enc)
        t_enc_out, t_attns = self.encoder(t_enc_out, attn_mask=None)

        s_dec_out = self.projection_s(s_enc_out).permute(0, 2, 1)[:, :, :N]
        t_dec_out = self.projection(t_enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        s_dec_out = s_dec_out * (s_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        s_dec_out = s_dec_out + (s_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        t_dec_out = t_dec_out * (t_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        t_dec_out = t_dec_out + (t_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # dec-> decomp
        x_dec_s, x_dec_t = self.decomposition(x_dec[:, -self.pred_len:, :])

        # t_dec_out -> decomp
        t_dec_out_s, t_dec_out_t = self.decomposition(t_dec_out)

        s_dec_out = s_dec_out + t_dec_out_s 
        t_dec_out = t_dec_out_t

        # out_bias = self.bias_adjustment(t_dec_out.permute(0,2,1)).permute(0,2,1)
        # t_dec_out = t_dec_out + 0.2*out_bias

        # Linear Regression 이용해서 추가
        X = np.array([[t] for t in range(-self.seq_len, 0)]) # -seq_len ~-1
        X_new = np.array([[t] for t in range(self.seq_len)]) # -seq_len ~-1

        vals = [[self.linear_regression_lstsq(X, t_enc_copy[idx, :, var]) for var in range(N)] for idx in range(B)]
        lin_result = [[self.linear_predict(X_new, vals[idx][var]) for var in range(N)] for idx in range(B)]
        # print(np.array(lin_result).shape)
        lin_result = torch.Tensor(np.array(lin_result)).to(self.device).permute(0, 2, 1)

        t_dec_out = lin_result
        # t_dec_out = 0.5 * t_dec_out + 0.49 * lin_result
        # t_dec_out = 0.001 * t_dec_out + 0.999 * lin_result
        # t_dec_out = t_dec_out + 0.2* lin_result

        # epsilon = 1e-6
        # if not torch.all(torch.abs(x_dec) < epsilon):
        #     me_dec_out = SMAE(t_dec_out.detach().cpu().numpy(), x_dec_t.detach().cpu().numpy())

        # t_dec_out = t_dec_out - 0.04 * torch.ones_like(t_dec_out)

        # dec_out = s_dec_out + t_dec_out
        dec_out = s_dec_out + t_dec_out

        return dec_out, t_dec_out, s_dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'long_term_forecast_partial']:
            dec_out, t_dec_out, s_dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], t_dec_out[:, -self.pred_len:, :], s_dec_out[:, -self.pred_len:, :] # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
