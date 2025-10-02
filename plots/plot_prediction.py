import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils.setup import _to_device

def _to_device(batch, device):
    """배치를 지정된 device로 이동"""
    if isinstance(batch, (list, tuple)):
        return [_to_device(item, device) for item in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch

def plot_station_with_predictions(
    model, valid_loader, valid_df, device, input_len=96, output_len=4, start_idx=0,
    feature_idx=0, # 입력 x에서 타깃 피처의 인덱스
    y_scaler=None, # 타깃 스케일러(학습 시 fit된 객체)
    log_target=True, # 타깃 로그 변환 -> True
    max_plots=None, # 모든 station plot - None
    save_path=None # 저장 경로
):
    
    # 역변환 함수 (standard scaling + 로그)
    def _inv_transform_1d(arr_1d):
        """
        Function: _inv_transform_1d
            - 1D 배열에 대해 역변환 수행
            - StandardScaler 역변환 후 로그 역변환 (expm1)
        Parameters:
            - arr_1d: np.ndarray
                - 1D 배열 (power 예측값)
        Returns:
            - np.ndarray
                - 역변환된 1D 배열
        """
        # Standard scaling 역변환
        if y_scaler is not None:
            arr_1d = y_scaler.inverse_transform(arr_1d.reshape(-1, 1)).reshape(-1)

        # 로그 역변환
        if log_target:
            arr_1d = np.expm1(arr_1d)
        return arr_1d
    
    # DataFrame 전처리
    df = valid_df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    
    # 모델을 평가 모드로 전환
    model.eval()
    
    # station별 예측값을 저장할 딕셔너리
    predictions = {}
    
    # 모델에서 예측값 생성
    with torch.no_grad():
        for batch in valid_loader:
            # 배치를 디바이스로 이동
            xb, s_idx, meta, yb = _to_device(batch, device)
            
            # 예측 수행
            if hasattr(model, "forward") and model.forward.__code__.co_argcount >= 4: # meta 인자 지원 여부 확인
                yhat = model(xb, s_idx, meta)
            else:
                yhat = model(xb)
            
            # 텐서를 CPU numpy로 변환
            yhat_np = yhat.cpu().numpy() # (B, T_out) 또는 (B, T_out, 1)
            s_np = s_idx.cpu().numpy() if s_idx is not None else None # (B,) # station_idx
            
            # 배치 내 각 샘플 처리
            for i in range(yhat_np.shape[0]):
                # station id 결정 (0, 1, 2, ... 형식)
                st = int(s_np[i]) if s_np is not None else i
                
                # 예측값 추출 및 역변환
                y_pred = yhat_np[i].reshape(-1)  # shape: (T_out,)
                y_pred_rec = _inv_transform_1d(y_pred) # standard + 로그 역변환
                
                # station별로 첫 번째 예측값만 저장 (각 스테이션당 하나의 샘플)
                if st not in predictions:
                    predictions[st] = y_pred_rec
            
            # plot할 station 수 제한
            if max_plots is not None and len(predictions) >= max_plots:
                break
    
    # station ID 매칭 함수
    def _extract_station_number(station_id):
        """
        Function: _extract_station_number
            - 'station00', 'station01' 형식의 문자열에서 숫자 추출
            - 이미 숫자인 경우 그대로 반환
        Parameters:
            - station_id: str | int
                - 발전소 ID ('station00' 형식 또는 숫자)
        Returns:
            - int
                - 추출된 발전소 번호 (숫자)
        """
        if isinstance(station_id, str) and station_id.startswith('station'):
            return int(station_id.replace('station', ''))
        return station_id
    
    def _format_station_id(station_num):
        """숫자를 'station00' 형식으로 변환"""
        return f"station{station_num:02d}"

    # station 목록 결정 (문자열 형식의 Station_ID와 숫자 형식의 예측 station 매칭)
    df_stations = df["Station_ID"].unique() # 'station00', 'station01', ...
    df_station_nums = [_extract_station_number(sid) for sid in df_stations] # [0, 1, 2, ...]
    pred_stations = list(predictions.keys()) # [0, 1, 2, ...]
    
    # 공통 station 찾기 (숫자 기준으로) # 매칭 되는 스테이션만 플롯
    available_station_nums = set(df_station_nums) & set(pred_stations)
    
    if max_plots is not None: 
        available_station_nums = sorted(list(available_station_nums))[:max_plots]
    else:
        available_station_nums = sorted(list(available_station_nums))
    
    n = len(available_station_nums)

    if n == 0:
        print("예측값과 매칭되는 station이 없습니다.")
        print(f"DataFrame station: {sorted(df_stations)}")
        print(f"예측값 station: {sorted(pred_stations)}")
        return
    
    # 서브플롯 그리드 계산
    cols = 3 if n >= 5 else n # 3 열 고정 (5개 이상일 때)
    rows = math.ceil(n / cols)
    
    plt.figure(figsize=(5*cols, 3*rows)) # (가로, 세로)
    
    for idx, station_num in enumerate(available_station_nums, start=1):
        # DataFrame에서 해당 스테이션 데이터 선택 (문자열 형식으로 변환)
        station_id_str = _format_station_id(station_num)
        sub = df[df["Station_ID"] == station_id_str].sort_values("date_time").reset_index(drop=True)
        
        # 시작 위치에서 input_window + output_window 구간 자르기
        sel = sub.iloc[start_idx:start_idx + input_len + output_len]
        
        if len(sel) < input_len + output_len:
            continue  # 데이터 부족 시 스킵

        # input_window part / output_window part 분리
        input_part = sel.iloc[:input_len]
        output_part = sel.iloc[input_len:]
        
        # 예측값 가져오기
        y_pred_rec = predictions[station_num]
        
        # 예측값의 시간축 생성 (output 구간과 동일한 길이)
        pred_time = output_part["date_time"][:len(y_pred_rec)]
        
        # 서브플롯 생성
        ax = plt.subplot(rows, cols, idx)
        
        # 실제값 플롯 (파랑: Input, 주황: Output)
        ax.plot(input_part["date_time"], input_part["power"], 
                color='blue', label="Input (Actual)", linewidth=2)
        ax.plot(output_part["date_time"], output_part["power"], 
                color='orange', label="Output (Actual)", linewidth=2)
        
        # 예측값 플롯 (초록)
        ax.plot(pred_time, y_pred_rec, 
                color='green', linestyle='--', label="Predicted", 
                linewidth=2,  markersize=4)
        
        # 과거/미래 경계선
        ax.axvline(input_part["date_time"].iloc[-1], 
                   color="black", linestyle=":", alpha=0.7)
        
        # 제목 설정 (Station 00, Station 01 형식으로 표시)
        ax.set_title(f"Station {station_num:02d}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power")
        
        # x축 틱 회전 (시간 표시 개선)
        ax.tick_params(axis='x', rotation=45)
        
        # 첫 번째 서브플롯에만 범례 표시
        if idx == 1:
            ax.legend(loc="best")
        
        # 그리드 추가
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_Owaolabi_predictions_chained(
    cnn, lstm, loader, df, device,
    input_len=10, output_len=1, start_idx=0,
    std_scaler=None,  
    mm_scaler=None,      
    logged: bool = False, 
    view_days = 2, 
    tgt_station_name="station",
    save_path=None
):
    # 역변환 (std_scaler -> mm_scaler -> (log))
    def _inv_transform_1d(arr_1d):
        arr = arr_1d.reshape(-1, 1) # (N,) -> (N, 1)
        if std_scaler is not None:
            arr = std_scaler.inverse_transform(arr)
        if mm_scaler is not None:
            arr = mm_scaler.inverse_transform(arr)
        arr = arr.reshape(-1) # (N, 1) -> (N,)
        if logged:
            arr = np.expm1(arr) # log 역변환
        return arr
    
    if "date_time" in df.columns:
        df = df.sort_values("date_time").reset_index(drop=True)
        df["date_time"] = pd.to_datetime(df["date_time"])
    
    # 시작/종료 시점 계산
    if start_idx + input_len >= len(df):
        print(f"[warn] start_idx+input_len가 데이터 길이 초과")
        return

    start_time = df.loc[start_idx, "date_time"] # 시작 시점
    view_end_time = start_time + pd.Timedelta(days=view_days) # 시각화 종료 시점

    # 입력 구간, 출력 구간 나누기
    # 입력: [start_time, start_time + input_len]
    input_start = start_time
    input_end = df.loc[start_idx + input_len - 1, "date_time"]
    input_mask = (df["date_time"] >= input_start) & (df["date_time"] <= input_end)
    input_part = df.loc[input_mask, ["date_time", "power"]].copy()
    
    # 출력 실제값: [input_end, view_end_time] 범위
    out_mask = (df["date_time"] > input_end) & (df["date_time"] <= view_end_time)
    output_part = df.loc[out_mask, ["date_time", "power"]].copy()

    # 예측을 view_days 단위 만큼 수행
    cnn.eval()
    lstm.eval()

    # 필요한 샘플 인덱스들을 수집
    # df.iloc[i+input_len : i+input_len+output_len]의 date_time
    needed_indices = []
    i = start_idx
    while True:
        out_start_i = i + input_len # 출력 시작 인덱스
        out_end_i = i + input_len + output_len # 출력 종료 인덱스 (미포함)
        if out_start_i >= len(df):
            break
        t0 = df.loc[out_start_i, "date_time"]
        if t0 > view_end_time:
            break
        needed_indices.append(i)
        i += 1

    if len(needed_indices) == 0:
        print("[warn] 예측에 사용할 샘플 인덱스가 없습니다.")
        return

    # 한 번의 loader 순회로 needed_indices에 해당하는 예측만 뽑아오기
    seen = 0 # 지금까지 본 샘플 수 (글로벌 오프셋)
    pred_map = {} # 예측 결과를 시간별로 누적(겹치면 최신 예측으로 덮어쓰기)

    with torch.no_grad():
        for batch in loader:
            xb = batch[0]
            bs = xb.size(0)

            g0 = seen
            g1 = seen + bs

            to_pick = [idx for idx in needed_indices if g0 <= idx < g1]
            if to_pick:
                xb, s_idx, meta, yb = _to_device(batch, device)
                
                # CNN 입력 형태로 변환
                if xb.dim() == 3:
                    xb = xb.permute(0, 2, 1)
                elif xb.dim() == 2:
                    xb = xb.unsqueeze(1)

                # 배치 전체 forward
                cnn_out = cnn(xb) # [B, F, L]
                yhat = lstm(cnn_out) # [B, output_len] or [B,]
                yhat_np = yhat.detach().cpu().numpy() # (B, H)

                # 필요한 오프셋만 꺼내 시간축에 매핑
                for idx in to_pick:
                    off = idx - g0 # 배치 내 오프셋
                    pred_seq = yhat_np[off].reshape(-1)
                    pred_seq = _inv_transform_1d(pred_seq) # 역변환
                    
                    # pred_seq를 df의 시간축에 맞춰 매핑
                    s = idx + input_len
                    e = min(s + output_len, len(df))
                    times = df.loc[s:e-1, "date_time"].values

                    # view_end_time을 넘는 부분은 버림
                    for t, v in zip(times, pred_seq[:len(times)]):
                        if t <= view_end_time:
                            pred_map[pd.Timestamp(t)] = float(v)

            # seen 이동
            seen += bs
            
            # 모든 needed_indices를 소화했으면 종료
            if seen > max(needed_indices):
                break

    # pred_map -> 정렬된 시계열로 변환
    if len(pred_map) == 0:
        print("[warn] 예측 결과가 비어 있습니다.")
        return
    
    pred_series = pd.Series(pred_map).sort_index()
    pred_time = pred_series.index
    y_pred_rec = pred_series.values

    # ===================================================
    # 시각화
    plt.figure(figsize=(10, 4.8))
    plt.plot(input_part["date_time"], input_part["power"], label="Input (Actual)", linewidth=2)
    plt.plot(output_part["date_time"], output_part["power"], label="Output (Actual)", linewidth=2)
    plt.plot(pred_time, y_pred_rec, linestyle="--", label="Predicted (chained)", linewidth=2, marker="o", markersize=3)
    plt.axvline(input_part["date_time"].iloc[-1], linestyle=":", alpha=0.7)

    title = f"[{tgt_station_name}] PVPlant Power Forecast | In_window={input_len}, Out_window={output_len}"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.xlim([input_start, view_end_time])
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()