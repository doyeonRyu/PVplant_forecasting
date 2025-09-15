import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as skm

"""
배치 데이터를 장치에 맞게 변환하는 함수
"""
def _to_device(batch, device):
    """
    Function: _to_device
        - DataLoader에서 가져온 배치를 장치에 맞게 변환
        - 동일한 배치 형태 유지를 위해
    Parameters:
        - batch: tuple
            - (xb, s_idx, meta, yb) 형태의 배치 데이터
        - device: torch.device
            - 데이터를 이동시킬 장치 (cuda)
    Returns:
        - tuple
            - 장치로 이동된 (xb, s_idx, meta, yb)
    """
    if len(batch) == 2:
        xb, yb = batch
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        return xb, None, None, yb
    else: 
        xb, s_idx, meta, yb = batch
        xb = xb.to(device, non_blocking=True)
        s_idx = s_idx.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        return xb, s_idx, meta, yb
    
"""
모델 평가 함수
- valid 또는 test 데이터셋에 대한 손실 계산
"""
@torch.no_grad()
def evaluate(loader, model, criterion, device):
    """
    Function: evaluate
        - valid 또는 test 데이터셋에 대한 손실 계산
    Parameters:
        - loader: DataLoader
            - 평가할 데이터셋의 DataLoader
        - model: 평가할 모델
        - criterion: 손실 함수
        - device: torch.device
    Returns:
        - float
            - 전체 데이터셋에 대한 평균 손실
    """
    model.eval() # 모델 평가 모드로 전환
    total, n = 0.0, 0 # 손실 합계 및 샘플 수 초기화

    for batch in loader: # 배치 단위로 반복
        xb, s_idx, meta, yb = _to_device(batch, device)
        if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
            # 4개 이상이면 s_idx, meta도 전달 (4개 이상: self, x, station_idx, meta)
            yhat = model(xb, s_idx, meta)
        else: # 4개 미만이면 s_idx, meta는 None
            yhat = model(xb)
            
        # 타깃 차원 보정 (1D -> 2D)
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1: 
            yb = yb.unsqueeze(-1)
        
        bs = xb.size(0) # 배치 크기 
        total += criterion(yhat, yb).item() * bs # 배치 손실 합산
        n += bs # 샘플 수 누적
    return total / max(n, 1) # 평균 손실 반환

"""
모델 평가 지표 계산 함수
- MAE, RMSE, MAPE 계산

"""
def metrics(best_path, model, data_loader, y_scaler, device):
    """
    Function: metrics
        - 최고 성능 모델 평가 지표 계산 함수
        - valid, test 데이터셋의 MAE, RMSE, MAPE 계산
        - 타켓: 스케일, 로그 역변환 
    Parameters:
        - best_path: train 함수에서 저장한 최고 성능 모델 파일 경로
        - model: 평가할 모델 객체
        - data_loader: 평가할 데이터셋의 DataLoader (valid_loader 또는 test_loader)
        - y_scaler: 타겟 스케일러 객체 (fit된 상태)
        - device: torch.device(cuda)
    """
    # best model 로드
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()

    # 타겟 실제값, 예측값 저장용
    all_y, all_yhat = [], []

    with torch.no_grad():
        for batch in data_loader:
            xb, s_idx, meta, yb = _to_device(batch, device)

            # forward 인자 개수 확인
            if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
                # 4개 이상이면 s_idx, meta도 전달 (4개 이상: self, x, station_idx, meta)
                yhat = model(xb, s_idx, meta)
            else: # 4개 미만이면 s_idx, meta는 None
                yhat = model(xb)

            # 타깃 차원 보정 (1D -> 2D)
            if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
                yb = yb.unsqueeze(-1)

            # CPU로 옮겨서 numpy 변환
            all_y.append(yb.cpu().numpy())
            all_yhat.append(yhat.cpu().numpy())

    # 배열 합치기 (배치 단위 -> 전체 샘플 단위)
    all_y = np.concatenate(all_y)
    all_yhat = np.concatenate(all_yhat)

    # 역변환: 스케일 -> 로그
    all_y  = y_scaler.inverse_transform(all_y).reshape(-1)
    all_yhat = y_scaler.inverse_transform(all_yhat).reshape(-1)

    # 로그 되돌리기
    all_y = np.expm1(all_y)
    all_yhat = np.expm1(all_yhat)

    # 지표 계산
    mae = skm.mean_absolute_error(all_y, all_yhat)
    mse = skm.mean_squared_error(all_y, all_yhat)
    rmse = np.sqrt(mse)
    mape = skm.mean_absolute_percentage_error(all_y, all_yhat)

    return mae, rmse, mape