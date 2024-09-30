import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import * # 메트릭 찾기
import torch
import torch.nn as nn

# 학습 데이터 바탕
# iTransformer, lin96, lin24 최적화 계수 수동으로 찾기

# 우선 실행
find_path = 'long_term_forecast_iTransformer_Exchange_96_96_Mod-iTransformer_data-exchange_rate.csv_(96to96)_0(1727520193)'

res_path = './results/'

pred = 'pred.npy'
true = 'true.npy'
pred_lin = 'pred_lin.npy'
pred_ensemble = 'pred_ensemble.npy'

np_pred = np.load(f"{res_path}{find_path}/{pred}")
np_true = np.load(f"{res_path}{find_path}/{true}")
np_pred_lin = np.load(f"{res_path}{find_path}/{pred_lin}")
np_pred_ensemble = np.load(f"{res_path}{find_path}/{pred_ensemble}")

# 0.6~1에서 0.05 단위로 샅샅이 검색

coefs_list = [0.5 + 0.05*n for n in range(11)] # 0.6~1까지

metric_map = {}
metric_list = ["MSE", "MAE", "SMAE", "STD_RATIO", "DSLOPE"]
metric_func = {"MSE": MSE, "MAE": MAE, "SMAE": SMAE, "STD_RATIO": STD_RATIO, "DSLOPE": SLOPE_RATIO}

# metric_map 정의
for key in metric_list:
    metric_map[key] = []
    for coef in coefs_list:
        metric_map[key].append(metric_func[key](coef*np_pred + (1-coef)*np_pred_lin, np_true) )
        print("METRIC_MAP APPEND :", f"{key}, {coef}")
# metric map 함수 그리기
graphs_path = "./graphs"

# 확인 - 출력
xr = range(11)

for key in metric_list:
    print(metric_map[key])
    plt.plot(xr, metric_map[key], labels=f"{key}_metric")
    plt.title("Metric distribution")
    plt.savefig(f"{graphs_path}/{key}_distrb.png")