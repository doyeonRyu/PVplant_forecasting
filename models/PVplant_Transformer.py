import torch
import torch.nn as nn
import math

"""
위치 인코딩
    - Transformer 모델의 입력에 위치 정보를 추가
    - Transformer 모델에서는 순서 정보가 없기 때문에 위치 인코딩을 통해 시퀀스 내 위치 정보를 제공
    - sin/cos 함수를 사용하여 각 위치에 대해 고유한 벡터 생성
    - 입력 임베딩에 더해져 모델이 순서 정보를 학습할 수 있도록 도움
"""
class PositionalEncoding(nn.Module):
    """
    Class: PositionalEncoding
        - Transformer 모델의 위치 인코딩 구현
    """
    def __init__(self, d_model, max_len=500):
        """
        Function: __init__
            - 위치 인코딩 초기화
        Parameters:
            - self: 객체
            - d_model: 임베딩 차원
            - max_len: 최대 시퀀스 길이 (inpuut window보다 같거나 커야 함)
        Returns: None
        """
        super().__init__()
        # (max_len, d_model) 0으로 초기화
        pe = torch.zeros(max_len, d_model)
        # 위치 인덱스 [0..max_len-1], shape=(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 주기 스케일링 항, 짝수 인덱스만큼 생성
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 짝수 채널: sin, 홀수 채널: cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 버퍼로 등록(학습 파라미터 아님, 디바이스 이동/저장 포함)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape=(1, max_len, d_model)

    def forward(self, x):
        """
        Function: forward
            - 입력 임베딩에 위치 인코딩 더하기
            - (B, L, d_model) 임베딩에 (1, L, d_model) 위치 인코딩 더함
        Parameters:
            - self: 객체
            - x: 입력 임베딩, shape=(batch_size, seq_len, d_model)
        Returns:
            - 위치 인코딩이 더해진 임베딩, shape=(batch_size, seq_len, d_model)
        """
        # 시퀀스 길이만큼 잘라서 더해줌
        return x + self.pe[:, :x.size(1)]

"""
시계열 임베딩
    - 입력 시계열 데이터를 Transformer에 맞게 임베딩하는 모듈
    - 입력 피처 수 F → d_model로 선형 투영
    - 위치 인코딩 추가
    - 레이어 정규화 및 드롭아웃 적용
"""
class TimeSeriesEmbedding(nn.Module):
    """
    Class: TimeSeriesEmbedding
        - 시계열 입력 데이터를 Transformer에 맞게 임베딩하는 모듈
    """
    def __init__(self, input_dim, d_model, max_len, dropout=0.1, use_scale=True):
        """
        Function: __init__
            - 임베딩 모듈 초기화
        Parameters:
            - input_dim: 입력 차원
            - d_model: 임베딩 차원
            - max_len: 최대 시퀀스 길이
            - dropout: 드롭아웃 비율
            - use_scale: 스케일링 사용 여부
        Returns: None
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) # 입력 F → d_model로 선형 투영
        self.pos_encoder = PositionalEncoding(d_model, max_len) # 위치 인코딩 모듈
        self.norm = nn.LayerNorm(d_model) # 각 시점 벡터 정규화
        self.dropout = nn.Dropout(dropout) # 드롭아웃
        self.use_scale = use_scale # 스케일 사용 여부
        self.scale = math.sqrt(d_model) if use_scale else 1.0 # √d_model로 나눠 안정화

    def forward(self, x):
        """
        Function: forward
            - (B, L, F) 실수 시계열을 (B, L, d_model) 임베딩으로 투영 + 위치 인코딩, 정규화, 드롭아웃
        Parameters:
            - self: 객체
            - x: 입력 임베딩, shape=(batch_size, seq_len, input_dim)
        Returns:
            - 위치 인코딩이 더해진 임베딩, shape=(batch_size, seq_len, d_model)
        """
        # x: (B, L, F)
        B, L, _ = x.shape # 배치/길이 추출
        # 1) 선형 투영 후 스케일링
        x = self.input_proj(x) / self.scale # (B, L, d_model)
        # 2) 위치 인코딩 더하기
        x = self.pos_encoder(x) # (B, L, d_model)
        # 3) 레이어 정규화
        x = self.norm(x) # (B, L, d_model)
        # 4) 드롭아웃
        x = self.dropout(x) # (B, L, d_model)
        return x

"""
Transformer
    - station별 power 예측을 위한 Transformer 모델
    - LSTM 모델과 유사한 구조를 갖도록 설계
    - encoder-only Transformer로 시계열 인코딩
"""
class PVPlantTransformer(nn.Module):
    """
    Class: PVPlantTransformer
        - station별 power 예측을 위한 Transformer 모델
        1) Transformer (encoder-only) 로 시계열 인코딩
        2) 마지막 타임스텝 표현 + station embedding + 메타 concat
        3) P 헤드로 output_size 회귀 
    """
    def __init__(
        self,
        input_size,
        d_model,
        output_size,
        num_layers,
        num_stations,
        station_emb_dim=8,
        meta_dim=0,
        dropout=0.1,
        nhead=4,
        dim_feedforward=512,
        max_len=500,
        pool="last", # "last" or "mean" (대표 벡터 추출 방식)
    ):
        """
        Function: __init__
            - Transformer 모델 초기화
        Parameters:
            - input_size: 입력 피처 수 (torch_train_x.shape[-1])
            - d_model: Transformer 임베딩 차원, LSTM의 hidden_size 역할
            - output_size: 예측 변수 수 (power = 1, multi-step = H)
            - num_layers: Transformer 인코더 레이어 수
            - num_stations: station 개수
            - station_emb_dim: station 임베딩 차원
            - meta_dim: 메타 피처 차원 (없으면 0)
            - dropout: 드롭아웃 비율
            - nhead: 멀티헤드 수
            - dim_feedforward: FFN 내부 차원
            - max_len: 최대 시퀀스 길이
            - pool: 'last' 또는 'mean' (대표 벡터 추출 방식)
        Returns:
            - None
        """
        super().__init__()
        self.d_model = d_model # Transformer 차원
        self.output_size = output_size # 최종 예측 길이
        self.pool = pool # last 또는 mean

        # 1) 실수 시계열 임베딩
        self.embedding = TimeSeriesEmbedding( 
            input_dim=input_size,
            d_model=self.d_model,
            max_len=max_len,
            dropout=dropout,
            use_scale=True,
        )

        # 2) Transformer Encoder 스택
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, # 토큰(시점) 임베딩 차원
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # (B, L, d) 유지
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers) # 인코더 스택
        self.enc_norm = nn.LayerNorm(self.d_model) # 인코더 출력 정규화

        # 3) station 임베딩
        self.st_emb = nn.Embedding(num_stations, station_emb_dim)

        # 4) FC head 입력 크기 정의. d_model + station_emb + meta → output_size
        fusion_in = self.d_model + station_emb_dim + (meta_dim if meta_dim > 0 else 0)

        self.head = nn.Sequential( 
            nn.Linear(fusion_in, max(32, fusion_in // 2)),  # 차원 축소
            nn.ReLU(), # 비선형 활성화 함수
            nn.Dropout(dropout), # 드롭아웃
            nn.Linear(max(32, fusion_in // 2), output_size) # (B, output_size)
        ) # output: (B, output_size). 최종 출력

    def forward(self, x, station_idx, meta=None):
        """
        Function: forward
            - Transformer 모델 순전파
        Parameters:
            - x: 입력 시퀀스 (B, L, input_size)
            - station_idx: 스테이션 인덱스 (B,)
            - meta: 메타 피처 (B, meta_dim) or None
        Returns:
            - y_hat: 예측 출력 (B, output_size)
        """
        # x: (B, L, F), station_idx: (B,), meta: (B, M) or None
        B = x.size(0) # 배치 크기

        # 1) 시계열 임베딩
        h = self.embedding(x) # (B, L, d_model)

        # 2) 인코더 통과
        h = self.encoder(h) # (B, L, d_model)
        h = self.enc_norm(h) # (B, L, d_model)

        # 3) 대표 벡터 추출(last 또는 mean)
        if self.pool == "last":
            rep = h[:, -1, :] # (B, d_model)
        else:
            rep = h.mean(dim=1) # (B, d_model)

        # 4) station 임베딩
        e = self.st_emb(station_idx.long()) # (B, E)

        # 5) 메타 유무에 따라 concat. 대표 벡터 + station 임베딩 + 메타 피처 결합
        if meta is not None:
            z = torch.cat([rep, e, meta], dim=-1) # (B, d_model+E+M)
        else:
            z = torch.cat([rep, e], dim=-1) # (B, d_model+E)

        # 6) 예측 헤드
        y_hat = self.head(z) 
        return y_hat # 예측 출력 # (B, output_size)
