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
                dilation,
                padding,
                groups,
                bias,
                padding_mode,
                dropout
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
        super(PVPlantCNN, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=dilation,
                             padding=padding,
                             groups=groups,
                             bias=bias,
                             padding_mode=padding_mode)
        self.bn = nn.BatchNorm1d(out_channels) # 배치 정규화 
        self.relu = nn.ReLU() # 활성화 함수 ReLU
        self.dropout = nn.Dropout(dropout) # 드롭아웃
        # self.flatten = nn.Flatten() # 평탄화 레이어

    def forward(self, x):
        """
        Function: forward
            - CNN 모델 순전파
        Parameters:
            - x: 입력 시퀀스 [B, C_in, L_in]
        Return:
            - x [B, L_out, C_out]
                - LSTM 모델 입력값으로 쓰이기 위한 형태
        """
        # x; [B, C_in, L_in] (B: 배치 크기, C_in: 입력 채널 수, L_in: 시퀀스 길이)
        x = self.cnn(x) # [B, C_out, L_out]
        x = self.bn(x) # 배치 정규화
        x = self.relu(x) # 활성화 함수
        x = self.dropout(x) # 드롭아웃
        # x = self.flatten(x) # 평탄화
        x = x.permute(0, 2, 1) # [B, C_out, L_out] -> [B, L_out, C_out] (LSTM에 맞춤)
        return x # [B, L_out, C_out]