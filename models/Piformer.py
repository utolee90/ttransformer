import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from sklearn.linear_model import LinearRegression


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
        self.d_model = configs.d_model

        self.joint_var = configs.joint_var # 패칭후 변수 상관관계 여부

        # get parameters
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)


        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seg_num_x, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
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
        # Decoder
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'long_term_forecast_partial']:
            self.projection = nn.Linear(configs.d_model, self.seg_num_y, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # linear_coef
        x_enc = x_enc.permute(0,2,1)
        # 1D convolution aggregation
        # x_enc = self.conv1d(x_enc.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x_enc
        x_enc = x_enc.permute(0,2,1)

        # downsampling: b,c,s -> b,c,n,w -> b,cw,n
        x_enc = x_enc.reshape(-1, self.enc_in,  self.seg_num_x, self.period_len)
        x_enc = x_enc.permute(0, 1, 3, 2).reshape(-1, self.enc_in*self.period_len, self.seg_num_x).permute(0,2,1)
        x_mark_enc = x_mark_enc[:, :self.seg_num_x, :]
        # print(x_enc.shape, x_mark_enc.shape)
        # x_mark_enc = x_mark_enc.reshape(-1, self.period_len*self.enc_in, self.seg_num_x).permute(0, 2, 1)

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # dec_out -> Projection -> composition
        dec_out = self.projection(enc_out) # shape bc, w, m
        dec_out = dec_out.reshape(-1, self.enc_in, self.period_len, self.seg_num_y )
        dec_out = dec_out.permute(0, 1, 3, 2).reshape(-1, self.enc_in, self.pred_len).permute(0,2,1)


        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

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
    
    # 부드럽게 Tensor 연결하는 함수 정의
    def smoothing(X):
        # X is a 4D tensor of shape (a, b, c, d)
        # weights is a 1D tensor of shape (2,)
        
        a, b, c, d = X.shape
        # 스택하고 리쉐이프하는 과정
        X_stacked = X.permute(1, 2, 0, 3).reshape(b, c, a * d)
        
        # 2개씩 짝을 지어 처리
        for j in range(a-1):  
            ct_point = d*(j+1)

            ct_before = X_stacked[:, :, ct_point-1]
            ct_after = X_stacked[:, :, ct_point]
            for k in range(d//2):
                X_stacked[:, :, k + ct_point - d//2] = (d-k)/(d+1) * X_stacked[:, :, k + ct_point - d//2] + (k+1)/(d+1) * ct_after
            for k in range(d//2):
                X_stacked[:, :, k + ct_point ] = (d//2 + k +1)/(d+1) * X_stacked[:, :, k + ct_point ] + (d//2 - k)/(d+1) * ct_before
        
        return X_stacked

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'long_term_forecast_partial']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
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
