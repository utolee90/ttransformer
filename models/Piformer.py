import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

model_segment = 'long_term_forecast_Exchange_24_24_iTransformer_iTransformer_custom_ftM_sl24_ll12_pl24_dm128_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0(1725429822)'


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 선형 레이어 정의 (y = Wx + b)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.base_model = configs.base_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len # 라벨 번호
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model

        self.joint_var = configs.joint_var # 패칭후 변수 상관관계 여부

        # get parameters
        self.enc_in = configs.enc_in # 변수 개수
        self.period_len = configs.period_len # 주기 단위

        self.seg_num_x = self.seq_len // self.period_len # 입력 조각
        self.seg_num_y = self.pred_len // self.period_len # 출력 조각

        # 새 인코더 차원 - 선형 
        self.new_encoders = nn.ModuleList()
        for i in range(self.seg_num_y):
            self.new_encoders.append(nn.Linear(self.seg_num_x,1).to('cuda:0'))


    def forecast(self, x_enc,  x_mark_enc, x_dec,  x_mark_dec, base_model):
        # Normalization from Non-stationary Transformer
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_dec = x_dec.clone() - means[0, 0, 0] * torch.ones_like(x_dec) # x_dec도 같이 표준화
        x_dec = x_dec.clone() / stdev[0, 0, 0] * torch.ones_like(x_dec) 

        _, _, N = x_enc.shape 

        # split x_enc into 
        # print(x_enc.shape)
        with torch.no_grad():
            x_enc_list = [x_enc[:,j*self.period_len:(j+1)*self.period_len, :] for j in range(self.seg_num_x)] # 입력값 나누기
            x_mark_enc_list = [x_mark_enc[:,j*self.period_len:(j+1)*self.period_len, :] for j in range(self.seg_num_x)] # 입력값 나누기
            x_dec = x_dec[:,-self.pred_len:, :] # dec 부분의 뒷부분만 사용
            x_mark_dec = x_mark_dec[:,-self.pred_len:, :] # 
            x_dec_list = [x_dec[:,j*self.period_len:(j+1)*self.period_len, :] for j in range(self.seg_num_y)] # 출력값 나누기
            x_mark_dec_list = [x_mark_dec[:,j*self.period_len:(j+1)*self.period_len, :] for j in range(self.seg_num_y)] # 출력값 나누기
        x_res1 = [] # 1단계 encoding
        x_res = [[] for _ in range(self.seg_num_y)] # n단계 encoding

        # 1단계 텐서 입력
        for k, x_enc_part in enumerate(x_enc_list):
            temp_enc = x_enc_part.clone() # 일단 복제
            # k -> x_seg_num - k번 반복
            for u in range(self.seg_num_x - k):
                if any(substr in self.base_model for substr in {'SparseTSF', 'PITS'}):
                    temp_enc = base_model(temp_enc)
                else:
                    # 0텐서 사용
                    temp_y_enc = torch.zeros(x_enc_part.shape[0], x_enc_part.shape[1]*3//2, x_enc_part.shape[2])
                    temp_enc = base_model(temp_enc, x_mark_enc_list[0], temp_y_enc, x_mark_dec_list[0])
            x_res1.append(temp_enc)
        
        # N단계 텐서 입력 -> x_res 채우기
        x_res[0] = x_res1
        for l in range(1, self.seg_num_y):
            if any(substr in self.base_model for substr in {'SparseTSF', 'PITS'}):
                x_res[l] = [base_model(enc_part) for enc_part in x_res[l-1]]
            else:
                temp_y_enc = torch.zeros(x_enc_part.shape[0], x_enc_part.shape[1]*3//2, x_enc_part.shape[2])
                # x_res[l] = [base_model(enc_part, x_mark_enc_list[0], temp_y_enc, x_mark_dec_list[0]) for enc_part in x_res[l-1]]
                x_res[l] = [base_model(enc_part.clone(), x_mark_enc_list[0], temp_y_enc.clone(), x_mark_dec_list[0]) for enc_part in x_res[l-1]]

        # 텐서 변환
        x_res_tensors = [torch.cat(encs).permute(1,2,0).reshape(-1,self.seg_num_x).to(x_enc.device) for encs in x_res]
        y_res_tensors = [decs.reshape(-1, 1) for decs in x_dec_list]

        # 훈련하기
        coefs = []
        for j in range(self.seg_num_y):
            y_res_tensors[j] = self.new_encoders[j](x_res_tensors[j])
            coef_ = self.new_encoders[j].weight[0]
            bias_ = self.new_encoders[j].bias
            coefs.append(torch.cat([coef_, bias_], axis=0))
        
        # y값 유도
        dec_out_stack = []
        for j, coef_vector in enumerate(coefs):

            dec_out_part = torch.zeros_like(x_dec_list[0])
            # print(x_dec_list[0].shape)
            
            for k, elem in enumerate(coef_vector):
                if k< self.seg_num_x:
                    # print(j, k, elem.item(), len(x_res[j]), x_res[j][k].shape)
                    dec_out_part = dec_out_part.clone() + elem.item() * x_res[j][k]
                else:
                    dec_out_part = dec_out_part.clone() + elem.item() * torch.ones_like(x_res[j][0])
            """
            if j == 0:
                for k, elem in enumerate(coef_vector):
                    if k< self.seg_num_x:
                        # print(j, k, elem.item(), len(x_res[j]), x_res[j][k].shape)
                        dec_out_part = dec_out_part.clone() + elem.item() * x_res[j][k]
                    else:
                        dec_out_part = dec_out_part.clone() + elem.item() * torch.ones_like(x_res[j][0])
            else:
                if any(substr in self.base_model for substr in {'SparseTSF', 'PITS'}):
                    dec_out_part = base_model(dec_out_stack[j-1])
                else:
                    # 0텐서 사용
                    temp_y_enc = torch.zeros(x_enc_part.shape[0], x_enc_part.shape[1]*3//2, x_enc_part.shape[2])
                    dec_out_part = base_model(dec_out_stack[j-1], x_mark_enc_list[0], temp_y_enc, x_mark_dec_list[0])
            """
            dec_out_stack.append(dec_out_part)

        # dec_out 값 유도
        # dummy coeff_vectors
        dummy_vectors = [[1/self.seg_num_x for _ in range(self.seg_num_x)]+ [0] for _ in range(self.seg_num_y)]
        dummy_vectors_2 = [[1 if j==self.seg_num_x-1 else 0 for j in range(self.seg_num_x +1)] for _ in range(self.seg_num_y)]

        dec_out = torch.cat(dec_out_stack, dim=1)

        # De-Normalization from Non-stationary Transformer

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
    

    def forward(self, x_enc,  x_mark_enc, x_dec, x_mark_dec, base_model):
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'long_term_forecast_partial']:
            dec_out = self.forecast(x_enc,  x_mark_enc, x_dec,  x_mark_dec, base_model)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
       
        return None

