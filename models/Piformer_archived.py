import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np




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

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.period_len, configs.d_model, configs.embed, configs.freq,
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
            self.projection = nn.Linear(configs.d_model, self.period_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

        
        

    def forecast(self, x_enc,  x_mark_enc, x_dec,  x_mark_dec, coeff_vectors, is_training):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_dec = x_dec - means[0, 0, 0] * torch.ones_like(x_dec)
        x_dec = x_dec / stdev[0, 0, 0] * torch.ones_like(x_dec)

        _, _, N = x_enc.shape 

        # split x_enc into  
        x_enc_list = [x_enc[:,j*self.period_len:(j+1)*self.period_len, :] for j in range(self.seg_num_x)] # 입력값 나누기
        x_dec = x_dec[:,-self.pred_len:, :] # dec 부분의 뒷부분만 사용
        # x_mark_dec = x_mark_dec[:,-self.pred_len:, :] # 
        x_dec_list = [x_dec[:,j*self.period_len:(j+1)*self.period_len, :] for j in range(self.seg_num_y)] # 출력값 나누기

        x_res1 = [] # 1단계 encoding
        x_res = [[] for _ in range(self.seg_num_x)] # n단계 encoding
        # 각 조각에 대해 조작
        for k, x_enc_part in enumerate(x_enc_list + x_dec_list):
            if k< len(x_enc_list + x_dec_list)-1:
                # Embedding
                # x_enc_part = x_enc_part.permute(0,2,1)
                enc_part_out = self.enc_embedding(x_enc_part, None)
                enc_part_out, attns = self.encoder(enc_part_out, attn_mask=None)

                #dec_out -> projection -> composition
                dec_part_out = self.projection(enc_part_out).permute(0, 2, 1)[:, :, :N]
                x_res1.append(dec_part_out)
 
        x_res[0] = x_res1

        # print("PART_TEST")
        # print(x_res1[0])
        # print(x_res1[2])

        # x_res 채우기
        for j in range(self.seg_num_x - 1):
            x_res_li = x_res[j]
            for k, x_enc_part in enumerate(x_res_li):
                if k< len(x_res_li)-1:
                    # Embedding
                    # x_enc_part = x_enc_part.permute(0,2,1)
                    enc_part_out = self.enc_embedding(x_enc_part, None)
                    enc_part_out, attns = self.encoder(enc_part_out, attn_mask=None)

                    #dec_out -> projection -> composition
                    dec_part_out = self.projection(enc_part_out).permute(0, 2, 1)[:, :, :N]
                    x_res[j+1].append(dec_part_out)
        
        # 훈련중일 때는 coeff_vectors 채우기
        lr = LinearRegressionModel(self.seg_num_x, 1).to(x_enc.device) # 차원,
        criterion = nn.MSELoss()
        optimizer = optim.SGD(lr.parameters(), lr=0.001)
        lr.train()

        if is_training:
            coef_col = []
            
            # Linear Regression 활용해서 보정.
            for k in range(self.seg_num_y):
                X_stack = torch.cat([
                    x_res[l][k + self.seg_num_x -1 - l].reshape(-1, 1) for l in range(self.seg_num_x) 
                ], axis=1)
                y = x_dec_list[k].reshape(-1, 1)
                # print("BEFORE LINEAR REGRESSION")
                # print(X_stack, y)

                
                # print(X_stack.shape)
                y_res = lr(X_stack)
                # print(coeff_vector)
                lin_loss = criterion(y_res, y)
                coeff_vector = torch.cat([lr.linear.weight[0], lr.linear.bias ], axis=0)
                # print(coeff_vector)
                # optimizer.zero_grad()
                # lin_loss.backward()
                # optimizer.step()
                
                # coeff_vector = lr_weight.data[0]
                coef_col.append(coeff_vector)
            
            coeff_vectors = torch.stack(coef_col)
            new_coeff_vector = torch.mean(torch.stack(coef_col), dim=0)
        
        # 나머지 coeff_vector 단순 torch 변환해서 사용
        else:
            # print("COEFF", coeff_vectors)
            coeff_vectors = torch.Tensor(coeff_vectors).to(x_enc.device)

        # dummy coeff_vectors
        dummy_vectors = [[1/self.seg_num_x for _ in range(self.seg_num_x)]+ [0] for _ in range(self.seg_num_y)]
        dummy_vectors_2 = [[1 if j==self.seg_num_x-1 else 0 for j in range(self.seg_num_x +1)] for _ in range(self.seg_num_y)]
        # coeff_vectors = torch.Tensor(dummy_vectors).to(x_enc.device)

        dec_out_stack = []
        dec_stack = x_enc_list.copy()

        # 훈련중 채우기
        if is_training:
            for j in range(self.seg_num_y):
                dec_out_part = torch.zeros_like(x_res[0][self.seg_num_x-1+j])
                for t in range(self.seg_num_x):
                    dec_out_part += coeff_vectors[j][-t-2]*x_res[t][self.seg_num_x-1-t+j]
                dec_out_part += coeff_vectors[j][-1]*torch.ones_like(x_res[0][self.seg_num_x-1+j])
                dec_out_stack.append(dec_out_part)
        
        # 테스트중 채우기 -> dec_stack 사용해서 정의 
        else:
            for j in range(self.seg_num_y):
                dec_out_part = torch.zeros_like(x_res[0][self.seg_num_x-1+j])
                for t in range(self.seg_num_x):
                    dec_out_part += coeff_vectors[j][-t-2]*dec_stack[-t-1]
                dec_out_part += coeff_vectors[j][-1]*torch.ones_like(x_res[0][self.seg_num_x-1+j])
                dec_out_stack.append(dec_out_part)
                dec_stack.append(dec_out_part)
        
        dec_out = torch.cat(dec_out_stack, dim=1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 최종적으로 coeff_vector -> numpy 형식으로 변환
        coeff_vectors = coeff_vectors.to('cpu').detach().numpy()

        return dec_out, coeff_vectors

    def imputation(self, x_enc,  x_mark_enc, x_dec, x_mark_dec, mask):
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

    def forward(self, x_enc,  x_mark_enc, x_dec, x_mark_dec, coeff_vectors=np.zeros(4), is_training=False, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'long_term_forecast_partial']:
            dec_out, coeff_vectors = self.forecast(x_enc,  x_mark_enc, x_dec,  x_mark_dec, coeff_vectors, is_training)
            return dec_out[:, -self.pred_len:, :], coeff_vectors  # [B, L, D]
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

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 선형 레이어 정의 (y = Wx + b)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
