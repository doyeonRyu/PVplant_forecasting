import torch
import torch.nn as nn

"""
1D CNN 모델
"""
class PVPlantCNN(nn.Module):
    """
    Class: PVPlantCNN
        - CNN + LSTM 융합 모델을 위한 1D CNN 모델
        1) 1D CNN으로 시계열 특징 추출
        2) LSTM에 맞게 차원 변환 (B, C_out, L_out) -> (B, L_out, C_out)
        3) LSTM에 전달
    """
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode='zeros',
                pool_kernel_size=2,
                pool_stride=2,
                dropout=0.0,
                lstm_in_features=32, # LSTM 입력 피처 수
                use_bn=True # 배치 정규화 사용 여부
    ):
        """
        Function: __init__
            - CNN 모델 초기화
        Parameters:
            - in_channels: input의 feature dimension
            - out_channels: output dimension
            - kernal_size: 한 kernel당 timestamp 개수
            - stride: kernel 이동 크기
            - dilation: kernel 내부에서 얼마만큼 띄어서 kernel을 적용할 것인가 (default: 1)
            - padding: 한 쪽 방향으로 얼마만큼 padding할 것인가 (그 만큼 양방향으로 적용) (default: 0)
            - groups: kernel의 height를 조절
            - bias: bias term을 둘 것인가 
            - padding_mode: 'zero', 'reflect', 'replicate', 'circular' (default: 'zero')
        Return: 
            - None
        """
        super().__init__()

        # 1D CNN 레이어
        self.cnn = nn.Conv1d(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            dilation,
                            padding,
                            groups,
                            bias,
                            padding_mode)
        
        # 배치 정규화 레이어 
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        # 활성화 함수 ReLU
        self.relu = nn.ReLU()
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        # max pooling 레이어 [B, C_out, L_conv] -> [B, C_out, L_pool]
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        # 평탄화 레이어 [B, C_out, L_pool] -> [B, C_out * L_pool]
        self.flatten = nn.Flatten()
        # LSTM 입력에 맞도록 Lazy Linear 레이어로 차원 맞추기
        self.fc = nn.LazyLinear(lstm_in_features)
    def forward(self, x):
        """
        Function: forward
            - CNN 모델 순전파
        Parameters:
            - x: 입력 시퀀스 [B, C_in, L_in]
        Return:
            - x: LSTM 입력에 맞게 변환된 시퀀스 [B, L_out=1, C_out=F]
            -  (B: 배치 크기, C_in: 입력 채널 수, L_in: 시퀀스 길이, C_out: 출력 채널 수, L_out: 출력 시퀀스 길이, F: LSTM 입력 피처 수)
        """
        # x: [B, C_in, L_in] (B: 배치 크기, C_in: 입력 채널 수, L_in: 시퀀스 길이)

        # Conv 블록 
        x = self.cnn(x) # [B, C_out, L_out]
        x = self.bn(x) # 배치 정규화 [B, C_out, L_out]
        x = self.relu(x) # 활성화 함수 [B, C_out, L_out]
        x = self.dropout(x) # 드롭아웃 [B, C_out, L_out]

        # Pooling + Flatten
        x = self.pool(x) # max pooling [B, C_out, L_pool]
        x = self.flatten(x) # 평탄화 [B, C_out * L_pool]

        # LSTM 입력에 맞게 차원 변환
        x = self.fc(x) # [B, F(lstm_in_features)]
        x = x.unsqueeze(1) # LSTM 입력에 맞게 차원 추가 [B, 1, F]
        return x # [B, L_out=1, C_out=F]
