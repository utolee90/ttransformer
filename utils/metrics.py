import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def SMAE(pred, true): # signed mae, 실제값 - 예측값
    return np.mean(true-pred)

# 상관계수 추측
def REC_CORR(pred, true, flag='mean'): 
    
    # 2차원 - 개별변수
    if pred.ndim == 2:
        if flag in  ['mean', 'average', 'me', 'avg']:
            return np.mean(np.array([np.corrcoef(pred[:,j], true[:,j])[0,1] for j in range(pred.shape[1]) if not np.isnan(np.corrcoef(pred[:,j], true[:,j])[0,1])]))
        elif flag in ['median', 'med']:
            return np.median(np.array([np.corrcoef(pred[:,j], true[:,j])[0,1] for j in range(pred.shape[1]) if not np.isnan(np.corrcoef(pred[:,j], true[:,j])[0,1])]))
        else:
            return 
    elif pred.ndim == 3:
        if flag == 'mean_total':
            return np.mean(np.array(
                    [
                        [np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for k in range(pred.shape[2]) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k])[0,1])] 
                        for j in range(len(pred))
                    ])
                )
        elif flag == 'median_total':
            return np.median(np.array(
                    [
                        [np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for k in range(pred.shape[2]) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k])[0,1])] 
                        for j in range(len(pred))
                    ])
                )
        elif flag in  ['mean', 'average', 'me', 'avg']:
            return np.mean([
                np.mean([np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k]))[0,1]])
                for k in range(pred.shape[2])
            ])
        elif flag in ['median', 'med']:
            return np.median([
                np.median([np.corrcoef(pred[j,:,k], true[j,:,k])[0,1] for j in range(len(pred)) if not np.isnan(np.corrcoef(pred[j,:,k], true[j,:,k]))[0,1]])
                for k in range(pred.shape[2])
            ])
        else:
            return 
 
def RATIO_IRR(pred, true, coef=2): # 오차값 분석. 기본값 표준편차 2
    
    tot_size = np.size(pred)
    mae = MAE(pred, true)

    # Calculate absolute errors
    err = np.abs(pred - true)
    
    # Determine large errors (errors that are k times larger than MAE)
    large_errors = err > coef * mae
    
    # Calculate the ratio of large errors
    large_error_ratio = np.sum(large_errors) / np.size(true)
    
    return large_error_ratio

# std_ratio - 참값과 예측값 표준편차 비교
def STD_RATIO(pred, true, flag='mean'):
    if pred.ndim == 2:
        if flag in  ['mean', 'average', 'me', 'avg']:
            std_pred = np.mean([np.std(pred[:, j]) for j in range(pred.shape[1])])
            std_true = np.mean([np.std(true[:, j]) for j in range(pred.shape[1])])
            return std_true / std_pred # 참값 대비 예측값 표준편차 평균
        elif flag in ['median', 'med']:
            std_pred = np.median([np.std(pred[:, j]) for j in range(pred.shape[1])])
            std_true = np.median([np.std(true[:, j]) for j in range(pred.shape[1])])
            return std_true / std_pred # 참값 대비 예측값 표준편차 평균
    elif pred.ndim == 3:
        if flag in  ['mean', 'average', 'me', 'avg']:
            std_pred = np.mean([np.mean([np.std(pred[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            std_true = np.mean([np.mean([np.std(true[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            return std_true / std_pred
        elif flag in  ['median', 'med']:
            std_pred = np.median([np.median([np.std(pred[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            std_true = np.median([np.median([np.std(true[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            return std_true / std_pred
    
    return None

# abs_ratio - mean 
def RATE_RATIO(pred, true, flag='mean'):
    X = np.array(range(pred.shape[-2])) # 기울기
    if pred.ndim == 2:
        if flag in  ['mean', 'average', 'me', 'avg']:
            rate_pred = np.mean([pred[-1,j] - pred[0,j] for j in range(pred.shape[1])])
            rate_true = np.mean([true[-1,j] - true[0,j] for j in range(pred.shape[1])])
            return len(X) *( rate_true - rate_pred ) # 참값 대비 예측값 평균기울기 평균
        elif flag in ['median', 'med']:
            rate_pred = np.median([pred[-1,j] - pred[0,j] for j in range(pred.shape[1])])
            rate_true = np.median([true[-1,j] - true[0,j] for j in range(pred.shape[1])])
            return len(X) * (rate_true - rate_pred) # 참값 대비 예측값 평균기울기 중간값
    elif pred.ndim == 3:
        if flag in  ['mean', 'average', 'me', 'avg']:
            rate_pred = np.mean([np.mean([pred[l, -1, j]-pred[l, 0, j] for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            rate_true = np.mean([np.mean([true[l, -1, j]-true[l, 0, j] for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            return len(X) * (rate_true - rate_pred)
        elif flag in  ['median', 'med']:
            rate_pred = np.median([np.median([pred[l, -1, j]-pred[l, 0, j] for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            rate_true = np.median([np.median([true[l, -1, j]-true[l, 0, j] for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            return len(X) * (rate_true - rate_pred)
    
    return None

# 선형회귀 계수 구하기
def get_slope(X, y):
    # X와 y의 평균
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # 기울기 (slope) 계산
    w = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
    
    return w

# slope_ratio - mean
def SLOPE_RATIO(pred, true, flag='mean'):
    X = np.array(range(pred.shape[-2])) # 기울기
    if pred.ndim == 2:
        if flag in  ['mean', 'average', 'me', 'avg']:
            rate_pred = np.mean([get_slope(X, pred[:, j]) for j in range(pred.shape[1])])
            rate_true = np.mean([get_slope(X, true[:, j])  for j in range(pred.shape[1])])
            return len(X) * (rate_true - rate_pred) # 참값 대비 예측값 기울기 평균
        elif flag in ['median', 'med']:
            rate_pred = np.median([get_slope(X, pred[:, j]) for j in range(pred.shape[1])])
            rate_true = np.median([get_slope(X, true[:, j])for j in range(pred.shape[1])])
            return len(X) * (rate_true - rate_pred) # 참값 대비 예측값 기울기 중간값
    elif pred.ndim == 3:
        if flag in  ['mean', 'average', 'me', 'avg']:
            rate_pred = np.mean([np.mean([get_slope(X, pred[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            rate_true = np.mean([np.mean([get_slope(X, true[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            return len(X)* (rate_true - rate_pred)
        elif flag in  ['median', 'med']:
            rate_pred = np.median([np.median([get_slope(X, pred[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            rate_true = np.median([np.median([get_slope(X, pred[l, :, j]) for j in range(pred.shape[2])]) for l in range(pred.shape[0])])
            return len(X)* (rate_true - rate_pred)
    
    return None