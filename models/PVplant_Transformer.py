import torch
import torch.nn as nn
import math

# ---------------------------------------------
# 위치 인코딩 (그대로 사용)
# ---------------------------------------------
class PositionalEncoding(nn.Module):
    '''
    기능: 위치 정보를 sin/cos로 인코딩하여 임베딩 벡터에 더함
    입력값: x (tensor, shape=[B, L, d_model])
    출력값: 위치 인코딩이 더해진 텐서 (동일 shape)
    '''
    def __init__(self, d_model, max_len=500):
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
        # 시퀀스 길이만큼 잘라서 더해줌
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------
# 실수 시계열 → d_model 임베딩 + 위치인코딩
# ---------------------------------------------
class TimeSeriesEmbedding(nn.Module):
    '''
    기능: (B,L,F) 실수 시계열을 (B,L,d_model) 임베딩으로 투영 + 위치 인코딩, 정규화, 드롭아웃
    입력값: x (tensor, shape=[B, L, input_dim])
    출력값: 임베딩 결과 (tensor, shape=[B, L, d_model])
    '''
    def __init__(self, input_dim, d_model, max_len, dropout=0.1, use_scale=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)     # 입력 F → d_model로 선형 투영
        self.pos_encoder = PositionalEncoding(d_model, max_len)  # 위치 인코딩 모듈
        self.norm = nn.LayerNorm(d_model)                   # 각 시점 벡터 정규화
        self.dropout = nn.Dropout(dropout)                  # 드롭아웃
        self.use_scale = use_scale                          # 스케일 사용 여부
        self.scale = math.sqrt(d_model) if use_scale else 1.0  # √d_model로 나눠 안정화

    def forward(self, x):
        # x: (B, L, F)
        B, L, _ = x.shape                                   # 배치/길이 추출
        # 1) 선형 투영 후 스케일링
        x = self.input_proj(x) / self.scale                 # (B, L, d_model)
        # 2) 위치 인코딩 더하기
        x = self.pos_encoder(x)                             # (B, L, d_model)
        # 3) 레이어 정규화
        x = self.norm(x)                                    # (B, L, d_model)
        # 4) 드롭아웃
        x = self.dropout(x)                                 # (B, L, d_model)
        return x


# ---------------------------------------------
# LSTMWithStationMeta와 동일한 인터페이스의 Transformer 버전
# ---------------------------------------------
class TransformerWithStationMeta(nn.Module):
    '''
    기능: LSTMWithStationMeta와 동일한 인터페이스를 갖는 인코더 전용 Transformer 모델
         - 입력 시계열을 Transformer Encoder로 인코딩
         - 마지막(or 평균) 타임스텝 표현 + 지점 임베딩(+옵션 메타) concat
         - MLP 헤드로 output_size(=horizon) 회귀
    입력값:
        input_size (int): 입력 피처 수(F)
        hidden_size (int): Transformer의 d_model로 사용 (호환성을 위해 이름만 동일)
        output_size (int): 예측 스텝 수(H), 최종 출력 차원
        num_layers (int): 인코더 레이어 수
        bidirectional (bool): LSTM 호환용 매개변수(Transformer에선 미사용)
        num_stations (int): 스테이션 개수 (nn.Embedding 용)
        station_emb_dim (int): 스테이션 임베딩 차원
        meta_dim (int): 메타 피처 차원(없으면 0)
        dropout (float): 드롭아웃 비율
        nhead (int): 멀티헤드 수 (기본 4)
        dim_feedforward (int): FFN 차원 (기본 512)
        max_len (int): 최대 시퀀스 길이
        pool (str): 'last' 또는 'mean' (대표 벡터 추출 방식)
    출력값:
        y_hat (tensor): shape=(B, output_size)
    '''
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        bidirectional=False,     # 호환성만 유지, 사용하지 않음
        num_stations=None,
        station_emb_dim=8,
        meta_dim=0,
        dropout=0.0,
        nhead=4,
        dim_feedforward=512,
        max_len=500,
        pool="last",
    ):
        super().__init__()
        # d_model로 hidden_size를 그대로 사용(호출부 변경 최소화)
        self.d_model = hidden_size                          # Transformer 차원
        self.output_size = output_size                      # 최종 예측 길이
        self.pool = pool                                    # 'last' 또는 'mean'

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
            d_model=self.d_model,                           # 토큰(시점) 임베딩 차원
            nhead=nhead,                                    # 멀티헤드 수
            dim_feedforward=dim_feedforward,                # FFN 내부 차원
            dropout=dropout,
            batch_first=True,                               # (B, L, d) 유지
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(self.d_model)          # 인코더 출력 정규화

        # 3) 스테이션 임베딩
        assert num_stations is not None, "num_stations를 지정하세요."
        self.st_emb = nn.Embedding(num_stations, station_emb_dim)

        # 4) Fusion(인코더 대표벡터 + station_emb + meta) → FC 헤드
        fusion_in = self.d_model + station_emb_dim + (meta_dim if meta_dim > 0 else 0)

        self.head = nn.Sequential(
            nn.Linear(fusion_in, max(32, fusion_in // 2)),  # 축소
            nn.ReLU(),                                      # 비선형
            nn.Dropout(dropout),                            # 드롭아웃
            nn.Linear(max(32, fusion_in // 2), output_size) # (B, output_size)
        )

    def forward(self, x, station_idx, meta=None):
        # x: (B, L, F), station_idx: (B,), meta: (B, M) or None
        B = x.size(0)                                       # 배치 크기
        # 1) 시계열 임베딩
        h = self.embedding(x)                               # (B, L, d_model)
        # 2) 인코더 통과
        h = self.encoder(h)                                 # (B, L, d_model)
        h = self.enc_norm(h)                                # (B, L, d_model)

        # 3) 대표 벡터 추출(last 또는 mean)
        if self.pool == "last":
            rep = h[:, -1, :]                               # (B, d_model)
        else:
            rep = h.mean(dim=1)                             # (B, d_model)

        # 4) 스테이션 임베딩
        e = self.st_emb(station_idx.long())                 # (B, E)

        # 5) 메타 유무에 따라 concat
        if meta is not None:
            z = torch.cat([rep, e, meta], dim=-1)           # (B, d_model+E+M)
        else:
            z = torch.cat([rep, e], dim=-1)                 # (B, d_model+E)

        # 6) 예측 헤드
        y_hat = self.head(z)                                # (B, output_size)
        return y_hat
