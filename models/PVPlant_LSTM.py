# LSTM for PV Plant Power Prediction

import torch
import torch.nn as nn

class PVPlantLSTM(nn.Module):
    """
    Class: LSTMWithStationMeta
        - station별 power 예측을 위한 LSTM 모델
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size, 
        num_layers=1,
        bidirectional=False,
        num_stations=None, 
        station_emb_dim=8, 
        meta_dim=0, 
        dropout=0.1,
    ):
        """
        Function: __init__
            - LSTM 모델 초기화
        Parameters:
            - input_size: LSTM 입력 피처 수 (torch_train_x.shape[-1])
            - hidden_size: LSTM 은닉 상태 크기
            - output_size: 예측 변수 수 (power = 1, multi-step = H)
            - num_layers: LSTM 레이어 수
            - bidirectional: 양방향 LSTM 여부
            - num_stations: station 개수 
            - station_emb_dim: station 임베딩 차원
            - meta_dim: 메타 피처 차원 (없으면 0)
            - dropout: 드롭아웃 비율
        Returns:
            - None
        """
        super().__init__()
        self.hidden_size = hidden_size # LSTM hidden state 크기
        self.num_layers = num_layers # LSTM 레이어 수
        self.num_directions = 2 if bidirectional else 1 # 양방향 여부 (1: 단방향, 2: 양방향)

        self.lstm = nn.LSTM( # LSTM 레이어
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0 # LSTM dropout은 layer > 1에서만 적용
        )

        # station 임베딩 레이어 (num_stations가 None이면 생성하지 않음)
        if num_stations is not None:
            self.st_emb = nn.Embedding(num_stations, station_emb_dim)
            emb_dim = station_emb_dim
        else:
            self.st_emb = None
            emb_dim = 0  # 임베딩 없으면 0

        # FC head 입력 크기 정의: LSTM hidden 출력 + (station 임베딩 있으면) + (메타 피처)
        fusion_in = hidden_size * self.num_directions + emb_dim + (meta_dim if meta_dim > 0 else 0)

        # 최종 fully connected head
        self.head = nn.Sequential(
            nn.Linear(fusion_in, max(32, fusion_in // 2)), # 첫 번째 FC 레이어
            nn.ReLU(), # 활성화함수 ReLU
            nn.Dropout(dropout), # 드롭아웃
            nn.Linear(max(32, fusion_in // 2), output_size) # output_size: 1 또는 H
        ) # output: (B, output_size). 최종 출력

    def forward(self, x, station_idx=None, meta=None):
        """
        Function: forward
            - LSTM 모델 순전파
        Parameters:
            - x: 입력 시퀀스 (B, L, input_size) # B: 배치 크기, L: 시퀀스 길이, input_size: 입력 피처 수
            - station_idx: 각 샘플의 station 인덱스 (B,)
            - meta: 메타 피처 (B, meta_dim) or None
        Returns:
            - y_hat: 예측 출력 (B, output_size)
    
        """
        B = x.size(0) # 배치 크기
        # 초기 은닉 상태 및 셀 상태 설정 (0으로 초기화)
        h0 = torch.zeros(self.num_layers * self.num_directions, B, self.hidden_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros_like(h0)

        out, (h_n, c_n) = self.lstm(x, (h0, c0)) # 입력 x를 LSTM에 통과
        # out: 전체 시퀀스 출력, (B, L, H * num_directions)
        # h_n: 마지막 hidden states, (num_layers * num_directions, B, H)
        # c_n: 마지막 cell states, (num_layers * num_directions, B, H)
        
        h_last = h_n[-1] # 마지막 레이어의 hidden state만 추출 (B, H * num_directions)

        # e = self.st_emb(station_idx) # station_idx를 embedding vector로 변환 (B, station_emb_dim)
        # st_emb가 있을 때만 사용하고, station_idx가 None이면 0벡터 대체
        e = None
        # self.st_emb가 None이 아니어야만 임베딩 관련 로직을 실행
        if self.st_emb is not None: 
            
            # 1. station_idx가 입력된 경우: 실제 임베딩 벡터 생성
            if station_idx is not None:
                e = self.st_emb(station_idx.long()) # (B, station_emb_dim)
                
            # 2. station_idx는 None이지만, 모델이 임베딩을 지원하는 경우: 0 벡터 생성
            else:
                # self.st_emb가 None이 아니므로 .embedding_dim에 안전하게 접근 가능
                emb_dim = self.st_emb.embedding_dim 
                e = torch.zeros(B, emb_dim, device=x.device, dtype=x.dtype) # (B, station_emb_dim) 0벡터

        # meta 피처 처리
        m = meta.to(device=x.device, dtype=x.dtype) if meta is not None else None

        # 특징 결합
        feats = [h_last]    # 항상 포함
        if e is not None:
            feats.append(e)
        if m is not None:
            feats.append(m)

        z = torch.cat(feats, dim=-1)

        # FC Head 통과 후 최종 출력
        y_hat = self.head(z)
        return y_hat # (B, output_size) 