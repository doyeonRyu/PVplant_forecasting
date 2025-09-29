import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    
    elif len(batch) == 4:
        xb, s_idx, meta, yb = batch
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # s_idx와 meta가 None이 아닐 때만 to(device)
        if s_idx is not None:
            s_idx = s_idx.to(device, non_blocking=True)
        if meta is not None:
            meta = meta.to(device, non_blocking=True)

        return xb, s_idx, meta, yb
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

class SeqDataset(torch.utils.data.Dataset):
    """
    Class: SeqDataset
        - 시계열 입력(x), 타깃(y), 발전소 인덱스(s), 메타데이터(meta)를 포함하는 데이터셋
        - 하나의 샘플을 (x, s, meta, y) 형태의 묶음으로 반환
        - 이후 DataLoader로 감싸서 배치 단위로 모델에 공급
    Parameters:
        - x: 입력 시계열, shape=(N, L, F)
        - y: 타깃 값, shape=(N,) 또는 (N, T)
        - s: 발전소 인덱스, shape=(N,)
        - meta: 메타데이터, shape=(N, M) 또는 None
    Returns: None
    """
    def __init__(self, x, y, s, meta=None):
        self.x = x
        self.y = y

        if s is None:
            self.s = None
        else:
            self.s = s 
            if self.s.dtype != torch.long:
                self.s = self.s.long()

        if meta is None:
            self.meta = None
        else:
            self.meta = meta 
            if self.meta.dtype not in (torch.float32, torch.float64):
                self.meta = self.meta.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        xi = self.x[i]
        yi = self.y[i]

        # 둘 다 None -> 2-튜플
        if (self.s is None) and (self.meta is None):
            return xi, yi

        # s만 None -> (x, meta, y) 3-튜플
        if self.s is None:
            mi = self.meta[i]
            return xi, mi, yi

        # meta만 None -> (x, s, y) 3-튜플
        if self.meta is None:
            si = self.s[i]
            return xi, si, yi

        # 둘 다 존재 -> 4-튜플
        si = self.s[i]
        mi = self.meta[i]
        return xi, si, mi, yi

def train(loader, model, criterion, optimizer, device):
    """
    Function: train
        - 모델을 한 epoch 동안 학습
    Parameters:
        - loader: DataLoader
            - 학습 데이터 로더
        - model: 학습할 모델
        - criterion: 손실 함수
        - optimizer: 옵티마이저
        - device: torch.device
    Returns:
        - float
            - 평균 학습 손실
    """
    model.train() # 모델 학습 모드로 전환
    total, n = 0.0, 0 # 손실 합계와 샘플 수 초기화

    # 배치 단위로 학습 
    for batch in loader: 
        xb, s_idx, meta, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환

        # 모델의 foward 함수에서 입력 인자 개수에 따라 다르게 호출
        if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
            # 4개 이상이면 s_idx, meta도 전달 (4개 이상: self, x, station_idx, meta)
            yhat = model(xb, s_idx, meta)
        elif hasattr(model, "forward") and model.forward.__code__.co_argcount == 3:
            # 3개면 s_idx만 전달 (3개: self, x, station_idx)
            yhat = model(xb, s_idx)
        else: # 2개면 s_idx, meta는 None
            yhat = model(xb)

        # 타깃 차원 보정 (1D -> 2D)
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            yb = yb.unsqueeze(-1)

        loss = criterion(yhat, yb) # 손실 계산
        optimizer.zero_grad(set_to_none=True) # 옵티마이저 기울기 초기화
        loss.backward() # 역전파
        optimizer.step() # 파라미터 업데이트

        bs = xb.size(0) # 배치 크기
        total += loss.item() * bs # 손실 합계 갱신
        n += bs # 샘플 수 갱신
    return total / max(n, 1) # 평균 손실 반환

def train_randomSearch(loader, model, criterion, optimizer, device, grad_clip=None):
    """
    Function: train
        - 모델을 한 epoch 동안 학습
    Parameters:
        - loader: DataLoader
            - 학습 데이터 로더
        - model: 학습할 모델
        - criterion: 손실 함수
        - optimizer: 옵티마이저
        - device: torch.device
    Returns:
        - float
            - 평균 학습 손실
    """
    model.train() # 모델 학습 모드로 전환
    total, n = 0.0, 0 # 손실 합계와 샘플 수 초기화

    # 배치 단위로 학습 
    for batch in loader: 
        xb, s_idx, meta, yb = _to_device(batch, device) # 배치를 장치에 맞게 변환

        # 모델의 foward 함수에서 입력 인자 개수에 따라 다르게 호출
        if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4:
            # 4개 이상이면 s_idx, meta도 전달 (4개 이상: self, x, station_idx, meta)
            yhat = model(xb, s_idx, meta)
        else: # 4개 미만이면 s_idx, meta는 None 
            yhat = model(xb)

        # 타깃 차원 보정 (1D -> 2D)
        if yb.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            yb = yb.unsqueeze(-1)

        loss = criterion(yhat, yb) # 손실 계산
        optimizer.zero_grad(set_to_none=True) # 옵티마이저 기울기 초기화
        loss.backward() # 역전파

        # gradient clipping (너무 큰 기울기 방지)
        if (grad_clip is not None) and (grad_clip > 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step() # 파라미터 업데이트

        bs = xb.size(0) # 배치 크기
        total += loss.item() * bs # 손실 합계 갱신
        n += bs # 샘플 수 갱신
    return total / max(n, 1) # 평균 손실 반환
