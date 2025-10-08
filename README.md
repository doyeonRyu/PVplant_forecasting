# 태양광 발전소 발전량 예측 (LSTM, Transformer(encoder-only))

[Notion |  https://www.notion.so/ryudoyeon/PV_Plant-Power-26561f7ccc3f802d875af29b7a099e65?source=copy_link]

<aside>
📌

Owaolabi et al. (2025) 논문 버전

[Owaolabi et al. (2025) 논문 따라한 버전](https://www.notion.so/Owaolabi-et-al-2025-28661f7ccc3f8096bb46c274dd8e5523?pvs=21)

</aside>

# 1. Object Setting

<aside>

- **LSTM**과 **Transforme**r를 멀티 모델로 구현
- 결과: 한 station의 power 혹은 모든 station의 각각의 power
- 멀티 모델로 여러 station을 모두 학습하여 결과 냄
- 두 모델의 성능 비교 (Transformer)가 우세하도록….
</aside>

목표

- 태양광 발전소의 발전량(power)을 NWP + LMD 기상 데이터와 meta 데이터를 활용해 예측
- 글로벌 모델: 여러 발전소를 동시에 학습해 공통 패턴 + 개별 특성을 함께 반영
- 시계열 데이터: 15분 단위 **2018년 08월 15일 16:00:00 - 2019년 06월 13일 15:45:00 (station 마다 상이)**


# 2. Data Curation & 3. Data Inspection

- 2. Data Curation
    
    ### Metadata
    
    > 발전소 특성 정보 데이터
    > 
    1. 열 목록
        - '`Station_ID`', '`Capacity`', '`PV_Technology`', '`Panel_Size`', '`Module`',
        '`Inverters`', '`Layout`', '`Panel_Number`', '`Array_Tilt`', '`Pyranometer`',
        '`Longitude`', '`Latitude`'
        - 열 설명
            
            - Station_ID: 발전소의 고유 식별자   
            - Capacity: 발전소의 설치 용량 (태양광 시스템이 최대 출력할 수 있는 정격 전력)   
            - PV_Technology: 사용된 태양광 패널 기술 종류   
            - Panel_Size: 태양광 모듈 1장의 크기
            - Module: 사용된 태양광 모듈의 모델명 또는 제조사
            - Inverters: 사용된 인버터 종류 및 개수
            - Layout: 패널 배치 방식
            - Panel_Number: 설치된 패널의 총 개수
            - Array_Tilt: 패널의 기울기 각도
            - Pyranometer: 설치 여부 또는 모델명
            - Longtitude: 발전소 경도 좌표
            - Latitude: 발전소 위도 좌표
            
    2. 각 열 시각화
        - 수치형 변수 Bar Chart
            - '`Capacity`', '`Panel_Size`', '`Panel_Number`’
            
            ![image.png](attachment:434a57de-1b02-4cb2-bf4e-67ee1fcea803:image.png)
            
        - 범주형 변수
            - ‘Module’, ‘Layout’, '`PV_Technology`', '`Inverters`', '`Array_Tilt`', '`Pyranometer`’
            - `PV_Technology`
                - Poly-Si    9
                - Mono-Si    1
            - `Inverters`
                - products types: …    2
                - products types: …    2
                - products types: …    1
                - products types: …    1
                - products types: …    1
                - products types: …    1
                - products types: …    1
                - products types: …    1
            - `Array_Tilt`
                - South 33°    5
                - South 31°    2
                - South 29°    1
            - `Pyranometer`
                - GHI: …    7
                - GHI: …    1
                - GHI: …    1
                - GHI: …    1
            - Preprocessing에서 인코딩 혹은 임베딩 필요
        - 위/경도 - 지도 시각화
            
            ![image.png](attachment:79aa8734-ac87-435d-a9d3-326052b23def:image.png)
            
    
    - `Module`, `Layout`: 10개가 모두 다름 - 학습 시 제외함
    - 스케일링 or 지역 군집화
    - 비슷한 중국 지역 밀집도 확인
    1. 이상치 / 결측치 확인
        - 건물 특성 정보 데이터라 10행
        - 이상치 / 결측치 존재하지 않음
    
    ### Station data
    
    > 시계열 데이터
    > 
    
    > stations로 전체 데이터 합침, [Station_ID, date_time]으로 정렬
    > 
    1. 열 목록
        - '`date_time`', '`nwp_globalirrad`', '`nwp_directirrad`', '`nwp_temperature`',
        '`nwp_humidity`', '`nwp_windspeed`', '`nwp_winddirection`', '`nwp_pressure`',
        '`lmd_totalirrad`', '`lmd_diffuseirrad`', '`lmd_temperature`', '`lmd_pressure`',
        '`lmd_winddirection`', '`lmd_windspeed`', '`power`'
        - 열 설명
            
            - date_time: timstamp   
            **NWP (Numerical Weather Prediction, 수치예보 모델 출력 값)**
            - nwp_globalirrad: Global Irradiance (태양이 지표면에 도달하는 총 일사량 (W/m2))
            - nwp_directirrad: Direct Irradiance (태양에서 직접 도달하는 일사 성분 (W/m2))
            - nwp_temperature: 기온, NWP에서 예측된 값
            - nwp_humidity: 상대습도, NWP에서 예측된 값
            - nwp_windspeed: 풍속, NWP 예보 값
            - nwp_winddirection: 풍향, NWP 예보 값
            - nwp_pressure: 기압, NWP 예보 값   
            **LMD (Local Meteorological Data, 현장 기상 관측 데이터)**
            - lmd_totalirrad: 총 일사량(Global Irradiance), 현장 측정 값   
            - lmd_diffuseirrad: 산란일사량(Diffuse Irradiance), 구름/대기 산란을 통해 도달하는 성분
            - lmd_temperature: 기온, 현장 기상 센서 측정 값
            - lmd_pressure: 기압, 현장 기상 센서 측정 값
            - lmd_winddirection: 풍향, 현장 측정 값
            - lmd_windspeed: 풍속, 현장 측정 값   
            **발전소 성능**  
            - power: 실제 발전소의 전력 출력   
            
            2. 예측값: power
            3. 각 기상변수 vs power 시각화
            - 발전소별 데이터 분포 확인
            - 그래프
            - 히트맵 (기상 변수와 발전량 간의 기본 상관 관계 확인)
            4. 발전소별 데이터 개수 균형, 운영 기간 체크
            5. 이상치, 결측치 확인
            - 평균, 중앙값, 이상치 등등 확인
            
    2. 예측값: `power`
        
        ![image.png](attachment:3739c9eb-76d4-487e-8956-a8ddf98b7958:image.png)
        
        - 각 발전소마다 시계열 범위가 다름을 확인
        - 글로벌 모델 학습 시 각 발전소마다 가진 데이터 기간만큼 학습하도록 반영
    3. 각 기상변수 vs power 시각화
        1. 각 발전소마다 따로 시각화
            - 비교 불가
        2. 각 열마다 따로 시각화 
            
            ![image.png](attachment:92760caf-fff0-4850-a4a8-9ef3a53430c6:image.png)
            
        3. 각 열과 power 상관관계 히트맵
            - 예시: station00
                
                ![image.png](attachment:516a0e15-9041-497d-af19-5145c05e5bb1:image.png)
                
            - 기상 변수와 발전량 간의 기본 상관 관계 확인
            - 발전소 모두 비슷한 경향을 보임
            - 상관 관계가 낮은 변수들도 학습에 포함 - 비선형 관계나 변수 상호작용을 위해서
            - 그러나 모델 학습 후 Feature importance 확인해서 기여도가 낮을 시 제거
                - Shapley values / permutation importance / attention score로 변수 기여도 확인
    4. 발전소별 데이터 개수 균형, 운영 기간 체크 
        - 우선, 범위 합집합 모델 학습 예정
        - 글로벌 모델 시 각 발전소 시계열 데이터 범위마다 구별, 학습/검증/테스트 분할, 보간 등 전처리 계획
            
            ```python
            Station 00: 2018-08-15 16:00:00 to 2019-06-13 15:45:00, Total records: 28896
            Station 01: 2018-06-30 16:00:00 to 2019-06-13 15:45:00, Total records: 33408
            Station 02: 2018-07-22 16:00:00 to 2019-06-10 15:45:00, Total records: 30432
            Station 03: 2019-01-11 16:00:00 to 2019-06-13 15:45:00, Total records: 14688
            Station 04: 2018-06-30 16:00:00 to 2019-06-13 15:45:00, Total records: 33408
            Station 05: 2019-03-04 16:00:00 to 2019-06-13 15:45:00, Total records: 9696
            Station 06: 2018-07-13 16:00:00 to 2019-06-13 15:45:00, Total records: 31104
            Station 07: 2018-06-30 16:00:00 to 2019-06-13 15:45:00, Total records: 32928
            Station 08: 2018-06-30 16:00:00 to 2019-06-13 15:45:00, Total records: 33120
            Station 09: 2018-09-25 16:00:00 to 2019-06-13 15:45:00, Total records: 24288
            ```
            
    5. 이상치 / 결측치 확인
        - 결측치 없음
        - 이상치 BoxPlot
            - IQR or Z-score로 NaN 값 처리 후 제거 or 보정 예정
        - 이상치 큰 변수
            - nwp_globalirrad, nwp_directirrad, nwp_windspeed, nwp_windspeed, lmd_totalirrad, lmd_diffuseirrad, lmd_windspeed

# 3. Data Inspection

- 3. Data Inspection
    
    ### Metadata
    
    1. 이상치, 결측치 확인
        - 건물 특성 정보 데이터라 10행
        - 이상치 / 결측치 존재하지 않음
    2. 열 분석
        - 수치형 / 범주형 변수 존재
        - Preprocessing에서 임베딩 필요
    3. 위/경도
        - 스케일링 or 지역 군집화
            - 비슷한 중국 지역 밀집도 확인
            - 스케일링으로 변환
    
    ### station data
    
    1. 이상치, 결측치 확인
        - 평균, 중앙값, 이상치 등등 확인
    2. 열 분석
    3. 발전소별 데이터 분포 확인
    4. 기상 변수와 발전량 간의 기본 상관 관계 확인

# 4. Data Preprocessing

- 4. Data Preprocessing
    
    ### Metadata
    
    1. 수치형 변수 → 스케일링
        - num_cols = ['Capacity', 'Panel_Size', 'Panel_Number']
        - 이상치 / 결측치 존재하지 않음
        - 단순 스케일링만 진행
        - Capacity, Panel_Number: StandardScaler # 이상치 존재 가능성 있음
        - Panel_Size: MinMaxScaler # 값이 다 비슷함
    2. 범주형 변수 → one-hot 임베딩
        - cat_cols = ['PV_Technology', 'Inverters', 'Array_Tilt', 'Pyranometer']
        - 데이터가 10개 밖에 안 되므로 전부 one-hot encoding
    3. 위치 → 스케일링 or 지역 군집화
        - 위, 경도
            - 중국 남동쪽에 몰려있음 - 단순 minmax scaling
    
    ### Station data
    
    1. 발전소별 정렬 및 시간 동기화
    2. Feature engineering
        - 날짜 관련 정보 추가
    3. train / valid / test 분할
        - 모든 경계는 월요일 - 일요일
        - 각 station마다 기간이 다르므로 비율로 환산해서 추출
        - 과정
            1. 해당 station의 전체 timestamp 길이 구하기   
            2. train: 길이의 80%, valid: 다음 10%, test: 마지막 10%로 분할   
            3. 이 때, 각 구간의 시작 -> 바로 다음 월요일로 변경    
            4. 각 구간의 마지막 -> 이전 일요일로 변경   
            5. gap: train -7일, valid - 7일로 gap 설정    
    4. X, y 분할
        - y는 추후 loss 확인에서만 사용할 것이므로 미리 분할
    5. 이상치 확인 후 결측치로 변환
        - 수치형 변수 IQR 1.5 초과, 미만 결측치로 변환
    6. 결측지 확인 후 보정
        - train
            - 보간, 앞 뒤
        - vali, test:
            - 앞, train 평균으로 나머지 채우기
    7. 풍향, 시간 → sin/cos 변환
    8. 타겟 로그 변환
    9. 스케일링
        - train으로만 fit → valid, test transform
    10. station_ID 임베딩하기 
        - embedding_dim = 8
    11. 슬라이딩 윈도우 생성 
        - input_window: 96
        - output_window: 4
    12. 메타데이터 준비
        - 메타데이터를 모델 입력 형태로 변환

# 5. Data Analysis (Modeling)

![image.png](attachment:8f85d20c-9a6e-40e9-b062-e316694f0030:2172a36f-f809-4a34-9cc6-2c79e28b17af.png)

1. 모델 초기화

- LSTM
- Transformer
- 마지막 hidden state와 meta 데이터 결합해 학습시킴
1. Seqdataset
    - 각 데이터를 (x, y, d, meta) 튜플로 각 샘플마다 분할
2. Dataloader
    - 샘플을 배치 사이즈에 따라 합쳐서 모델 입력을 위한 최종 형태 반환
3. 모델 train
4. 모델 validation data 평가 지표
    - MAE, RMSE, MAPE
    

모델 별 파라미터 설정

1. LSTM 
    1. 기본 세팅 값
        
        <aside>
        
        input_size  = torch_train_x.shape[-1] *# 22*
        
        output_size = torch_train_y.shape[-1] if torch_train_y.dim() == 2 else 1
        
        num_stations = len(station_to_idx)
        
        num_layers = 2
        
        meta_dim = torch_train_meta.shape[1] if 'torch_train_meta' in globals() and torch_train_meta is not None else 0
        
        station_emb_dim = 8
        
        dropout = 0.2
        
        hidden_size = 128
        
        d_model = 128
        
        nhead = 8
        
        dim_feedforward = 512
        
        num_ts_layers = 3
        
        </aside>
        
    2. randomized search 버전
        
        <aside>
        
        input_size  = 22
        output_size = 672
        num_stations = 10 # len(station_to_idx)
        num_layers = 1
        meta_dim = 20
        station_emb_dim = 12
        dropout = 0.1
        hidden_size = 96
        
        epochs = 100
        patience = 15
        grad_clip = 0.5
        optimizer = Nadam
        loss = Huber
        Ir = 0.0005
        Weight_decay = 0.0
        
        input_window: 1달
        output_window: 1주일
        
        </aside>
        
2. Transformer
    1. 기본 세팅 값
        
        <aside>
        
        input_size  = torch_train_x.shape[-1] *# 22*
        
        output_size = torch_train_y.shape[-1] if torch_train_y.dim() == 2 else 1
        
        num_stations = len(station_to_idx)
        
        num_layers = 2
        
        meta_dim = torch_train_meta.shape[1] if 'torch_train_meta' in globals() and torch_train_meta is not None else 0
        
        station_emb_dim = 8
        
        dropout = 0.2
        
        hidden_size = 128
        
        d_model = 128
        
        nhead = 8
        
        dim_feedforward = 512
        
        num_ts_layers = 3
        
        </aside>
        
    2. randomized search 버전
        
        <aside>
        
        input_size  = 22
        output_size = 672
        num_stations = 10 # len(station_to_idx)
        meta_dim = 20
        station_emb_dim = 8
        dropout = 0.2
        
        d_model = 128
        nhead = 8
        dim_feedforward = 512
        num_ts_layers = 3
        
        epochs = 20
        patience = 15
        grad_clip = 0.5
        optimizer = Adam
        loss = Huber
        Ir = 0.001
        Weight_decay = 0.0
        
        input_window: 1달
        output_window: 1주일
        
        </aside>
        

## 결과

|  | **기본 세팅 값** | **randomized search 버전** |
| --- | --- | --- |
| **LSTM**  | MAE: 0.5440
RMSE: 1.2909
MAPE: 5.9687 | MAE: 1.0655
RMSE: 2.1215
MAPE: 257.1057 |
| **Transformer** | MAE: 0.5440
RMSE: 1.2909
MAPE: 5.9687 | MAE: 1.3821
RMSE: 2.5282
MAPE: 209.2254 |

# 6. Deployment

- 발전소별 예측값 vs 실제값 시각화
    1. Output_window가 여러 값일 때 
        - plot_station_with_predictions 함수 사용
        - LSTM
            
            ![image.png](attachment:b5fc3b3f-3f7d-4ad5-a2a2-d6df92b8fbff:image.png)
            
        - Transformer
            
            ![image.png](attachment:9db77eb7-8c2f-4647-b7ae-8621d8e00058:image.png)
            
    2. Output_window가 하나일 때 
        - plot_station_forecast_chained 함수 사용
        - 기존 함수로는 선 그래프 적용 불가
        - 여러 예측값을 잇는 형태로 변환
        - station별 시간축 밀림 현상 해결 못 함
        - LSTM
            
            ![image.png](attachment:052fb22c-626c-46e5-aef1-ba192222de89:image.png)
            
        - Transformer
            
            ![image.png](attachment:e89bf635-fd36-408a-9c33-6e31d98e6247:image.png)