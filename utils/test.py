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
def evaluate(loader, model, criterion, device):
    '''
    함수 설명:
        - valid/test DataLoader에 대해 전체 평균 손실과 station별 평균 손실을 계산함
    입력값:
        - loader: torch.utils.data.DataLoader, (xb, s_idx, meta, yb) 형태 배치를 제공
        - model: torch.nn.Module, forward(x[, station_idx, meta]) 지원
        - criterion: 손실 함수(일반적으로 reduction='mean')
        - device: torch.device, 연산에 사용할 디바이스
    출력값:
        - (overall_loss: float, station_avg: dict[int, float])
          overall_loss는 전체 평균 손실
          station_avg는 station_id -> 평균 손실
    '''
    model.eval() # 모델을 평가 모드로 전환
    total_sum = 0.0 # 전체 손실의 합(가중합)
    total_cnt = 0 # 전체 샘플 수

    from collections import defaultdict # station별 합/개수 저장을 위해 defaultdict 사용
    st_sum = defaultdict(float) # station별 손실 합
    st_cnt = defaultdict(int) # station별 샘플 수

    # 그래디언트 비활성화(평가 단계이므로 메모리/속도 이점)
    with torch.no_grad():  # torch를 이미 임포트했다고 가정
        for batch in loader:  # 배치 단위로 반복
            xb, s_idx, meta, yb = _to_device(batch, device)  # 배치를 디바이스로 이동

            # 모델 인자 수에 따라 station/meta 전달 여부 결정
            if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
                # forward(self, x, station_idx, meta) 형태
                yhat = model(xb, s_idx, meta)  # station/meta까지 포함하여 예측
            else:
                # forward(self, x) 형태
                yhat = model(xb)  # 입력 시계열만으로 예측

            # 타깃 차원 보정: (N,) vs (N,1) 불일치 시 정렬
            if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
                yb = yb.unsqueeze(-1)  # 타깃을 (N,1)로 확장

            bs = xb.size(0)  # 현재 배치 크기
            # criterion이 보통 배치 평균을 반환하므로, * bs로 가중합을 만든다
            batch_loss_mean = criterion(yhat, yb).item()  # 배치 평균 손실
            total_sum += batch_loss_mean * bs # 전체 손실 합에 가중합 추가
            total_cnt += bs # 전체 샘플 수 누적

            # station별 손실: 배치 내 station 고유값별로 슬라이스하여 동일 방식으로 합산
            # s_idx는 (N,) 또는 (N,1)일 수 있으므로 1D 텐서로 맞춘다
            if s_idx.dim() > 1:
                s_idx_flat = s_idx.view(-1) # (N,)로 평탄화
            else:
                s_idx_flat = s_idx # 이미 (N,)

            unique_stations = s_idx_flat.unique() # 배치에 등장한 station ID들
            for st in unique_stations: # 각 station별로 손실 계산
                mask = (s_idx_flat == st) # 해당 station의 마스크 (N,)
                idx = mask.nonzero(as_tuple=True)[0] # 해당 인덱스들
                # 해당 station에 속한 샘플만 슬라이스
                yhat_sub = yhat.index_select(0, idx)
                yb_sub = yb.index_select(0, idx)
                sub_cnt = yhat_sub.size(0) # 해당 station 샘플 수
                # 동일하게 평균 손실을 구하고, * sub_cnt로 가중합을 만든다
                st_loss_mean = criterion(yhat_sub, yb_sub).item()
                st_sum[int(st.item())] += st_loss_mean * sub_cnt # station 손실 합
                st_cnt[int(st.item())] += sub_cnt  # station 샘플 수

    # 전체 평균 손실 계산(샘플 수로 나눔)
    overall_loss = total_sum / max(total_cnt, 1)

    # station별 평균 손실 계산
    station_avg = {}
    for k in st_sum.keys():
        station_avg[k] = st_sum[k] / max(st_cnt[k], 1)

    return overall_loss, station_avg

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