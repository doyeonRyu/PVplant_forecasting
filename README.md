# PV Plant power 예측 모델 

참고    
> PVOD v1.0 : A photovoltaic power output dataset    
> 링크: https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62   

> [Notion 정리](https://www.notion.so/ryudoyeon/PV_Plant-Power-26561f7ccc3f802d875af29b7a099e65?source=copy_link)   

## 구현 방향성
- station 데이터를 모두 학습한 글로벌 모델 생성
- LSTM과 transformer 동시 구현을 통한 비교, 경쟁

### 1. Object Setting
> **LSTM**과 **Transforme**r를 멀티 모델로 구현  
> 결과: 한 station의 power 혹은 모든 station의 각각의 power  
> 멀티 모델로 여러 station을 모두 학습하여 결과 냄  
> 두 모델의 성능 비교 (Transformer)가 우세하도록….  

목표   
- 태양광 발전소의 발전량(power)을 NWP + LMD 기상 데이터와 meta 데이터를 활용해 예측
- 글로벌 모델: 여러 발전소를 동시에 학습해 공통 패턴 + 개별 특성을 함께 반영
- 시계열 데이터: 15분 단위 **2018년 08월 15일 16:00:00 - 2019년 06월 13일 15:45:00**

### 2. Data Curation & 3. Data Inspection
1. metadata  

2. station 데이터   

### 4. Data Preprocessing  
1. metadata  
    1) 수치형 변수 → 스케일링  
    2) 범주형 변수 → one-hot 임베딩  
    3) 위치 → 스케일링 or 지역 군집화  
    4) station_id 임베딩  

    - train/valid/test 분할 필요 없음  

2. station data
    1) NWP / LMD / Power 정렬 및 시간 동기화
    2) Feature engineering
        - 날짜 관련 정보 추가
    3) train / valid / test 분할
        - 모든 경계는 월요일 - 일요일    
        - 각 station마다 기간이 다르므로 비율로 환산해서 추출   

        과정   
            1. 해당 station의 전체 timestamp 길이 구하기   
            2. train: 길이의 80%, valid: 다음 10%, test: 마지막 10%로 분할   
            3. 이 때, 각 구간의 시작 -> 바로 다음 월요일로 변경    
            4. 각 구간의 마지막 -> 이전 일요일로 변경   
            5. gap: train -7일, valid - 7일로 gap 설정  
    4) X, y 분할             
    5) 이상치 확인 후 결측치로 변환
    6) 결측지 확인 후 보정
        - 보간, 앞 뒤
    7) 풍향, 시간 → sin/cos 변환
    8) 타겟 로그 변환
    9) 스케일링
        - train으로만 fit → valid, test transform
    10) Station_ID 임베딩하기
    11) 슬라이딩 윈도우 생성

### 5. Data Analysis (Modeling)
    1. 텐서로 변환
    2. Dataloader 생성
    3. 모델 생성
    4. 모델 학습 
        - 여러 발전소를 동시에 학습하여 공통 패턴과 발전소 특성을 함께 반영
        - 학습 시 모델을 통해 시계열 특성을 파악하고 마지막에 metadata 반영
    5. 모델 성능 계산
        - 각 station 별 power 예측
        - MAE, RMSE, MAPE
    6. validation
    7. 모델 예측
        - test 데이터 사용
        - 각 station 별 power 예측