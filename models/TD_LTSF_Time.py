import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

class Model(nn.Module):
    """
    take a + tb as the prediction. a and b are learnable parameters.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.channels = configs.enc_in
        self.individual = configs.individual

        FIBONACCI_NUMBERS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
        self.fibonacci_labels = [self.seq_len - 1 - f for f in FIBONACCI_NUMBERS if self.seq_len - 1 - f >= 0]
        self.last_labels = [self.seq_len - 1]

        self.coef = nn.ModuleList()
        for i in range(self.channels):
            linear_layer = nn.Linear(1 + len(self.fibonacci_labels), 2)
            init.kaiming_uniform_(linear_layer.weight, a=0.01)  # He 초기화
            init.constant_(linear_layer.bias, 0)  # 편향을 0으로 초기화
            self.coef.append(linear_layer)

    def forward(self, x):
        # x: [Batch, Input length, Channel]

        output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
        res = torch.zeros([x.size(0),2,x.size(2)],dtype=x.dtype).to(x.device) # [Batch, 2, Channel] coefficients

        fibonacci_labels = self.fibonacci_labels
        for i in range(self.channels):
            x_ = x[:,:,i]
            x_ = x_[:,[1] + fibonacci_labels]
            res[:,:,i] = self.coef[i](x_)
            output[:,:,i] = torch.stack([ res[:,0,i] + t*res[:,1,i]/(1+len(fibonacci_labels)) for t in range(self.pred_len)], dim=1)
        
        # print("TEST", res[0, 0, 0], res[0,1,0])
        # print("RES", output[0, :, 0])
        x = output

        return x # [Batch, Output length, Channel]