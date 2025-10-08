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

"""
각 station별 예측 결과 시각화 함수
"""
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

"""
각 station별 Output_window가 1일 때 연속적으로 예측 결과 시각화하는 함수 
"""
def plot_station_forecast_chained(
    model, data_loader, data_df, device,
    input_len=10, output_len=1, start_idx=0,
    y_scaler=None,
    log_target=True,
    station_id=None,
    view_days=2,
    max_plots=None,
    permute_to_BFL=False,
    save_path=None
):
    """
    Function: plot_station_with_predictions
        - 각 스테이션별로 예측 결과를 시각화
    Parameters:
        - model: best_lstm or best_transformer
        - data_loader: DataLoader (valid or test)
        - df: pd.DataFrame (valid_df or test_df)
        - device: cuda
        - input_len = input_window (default: 10)
        - output_len = output_window (default: 1)
        - start_idx: int (default: 0)
            - 각 스테이션별로 시작 인덱스 (로컬 인덱스)
        - y_scaler: StandardScaler (default: y_scaler)
        - log_target: bool (default: True)
            - 타겟이 로그 변환된 경우 True
        - station_id: str or list (default: None)
            - 특정 스테이션 ID 또는 ID 리스트 (None이면 첫 번째 스테이션만)
        - view_days: 시각화할 기간 (일 단위, default: 2)
        - max_plots: int or None (default: None)
            - 최대 시각화할 스테이션 수 (None이면 모두)
        - permute_to_BFL: bool (default: False)
            - 모델 입력 텐서의 차원 순서가 (B, F, L)인 경우 True
        - save_path: str or None (default: None)
    Returns:
        - None (시각화 출력 및 저장)
    """
    
    # ========== 역변환 함수 (standard scaler -> expm1)==========
    def _inv_standard_log1p(arr_1d: np.ndarray) -> np.ndarray:
        a = arr_1d.reshape(-1, 1) # 2D array로 변환
        if y_scaler is not None: # standard scaler 역변환
            a = y_scaler.inverse_transform(a)
        a = a.ravel() # 1D array로 변환
        if log_target: # expm1 역변환
            a = np.expm1(a)
            a = np.maximum(a, 0) # 음수 제거
        return a

    # ========== 데이터 준비 ==========
    df_all = data_df.copy()
    if "date_time" in df_all.columns: 
        df_all["date_time"] = pd.to_datetime(df_all["date_time"])
    
    df_all = df_all.reset_index(drop=True) 
    df_all['global_idx'] = df_all.index # 글로벌 인덱스 추가
    # global_idx example: station_01 (0~999), station_02 (1000~1999), ... 

    # station 리스트 준비
    if "Station_ID" in df_all.columns:
        all_stations = df_all["Station_ID"].astype(str).unique().tolist()
        #  all_stations example: ['station_01', 'station_02', ...]
    else:
        all_stations = [None]

    # station_id 정규화 | 리스트 형태로 변환
    if station_id is None:
        station_list = all_stations[:1]
    elif isinstance(station_id, (list, tuple, pd.Series, np.ndarray)):
        station_list = pd.Series(station_id, dtype="object").astype(str).unique().tolist()
        # station_list example: ['station_01', 'station_02', ...]
    else:
        station_list = [str(station_id)]

    # max_plots 적용: station_list 자르기
    if max_plots is not None: 
        station_list = station_list[:int(max_plots)] 

    # ========== 배치 언패킹 함수 ==========
    def _unpack_batch(batch):
        xb = batch[0]
        s_idx = None
        meta = None
        if len(batch) >= 2 and isinstance(batch[1], torch.Tensor):
            if len(batch) == 4:
                s_idx, meta = batch[1], batch[2]
            elif len(batch) == 3:
                s_idx = batch[1]
        return xb, s_idx, meta

    # ========= station별 예측 및 시각화 ==========
    for sid in station_list:
        print(f"\n{'='*60}")
        print(f"Processing Station: {sid}")
        print(f"{'='*60}")
        
        # station별 데이터 필터링
        if sid is None or "Station_ID" not in df_all.columns: # 단일 station 또는 ID 컬럼 없는 경우
            df = df_all.copy() # 전체 사용
        else: # 여러 station 중 sid에 해당하는 station 필터링
            df = df_all[df_all["Station_ID"].astype(str) == sid].copy()

        df = df.sort_values("date_time").reset_index(drop=True)
        
        if start_idx + input_len >= len(df):
            print(f"[경고] Station {sid}: 데이터 부족")
            continue

        # ========== 시간 구간 설정 ==========
        start_time = df.loc[start_idx, "date_time"] # input_window 시작 시점
        input_start = start_time 
        input_end = df.loc[start_idx + input_len - 1, "date_time"] # input_window 끝 시점
        view_end = start_time + pd.Timedelta(days=view_days) # 시각화 종료 시점

        input_mask = (df["date_time"] >= input_start) & (df["date_time"] <= input_end) # input_window 구간
        input_part = df.loc[input_mask, ["date_time", "power"]].copy() 
        
        output_mask = (df["date_time"] > input_end) & (df["date_time"] <= view_end) # 예측 확인용 실제값 필터
        output_part = df.loc[output_mask, ["date_time", "power"]].copy()

        # ========== 예측 인덱스 계산 ==========
        """
        핵심 개념:
        - input_window_start: 입력 윈도우의 시작 로컬 인덱스 (모델에 넣을 샘플)
        - pred_timestamp_idx: 예측값이 대응되는 실제 타임스탬프의 로컬 인덱스
        
        예시 (input_len=10, output_len=1):
        - 윈도우 [0~9] → 예측 시점 10
        - 윈도우 [1~10] → 예측 시점 11
        """
        # 예측 시점 정보를 저장할 리스트 초기화 
        prediction_info = []  # [(global_idx, input_start_local, pred_timestamp_local), ...]
        
        # 입력 윈도우 시작 인덱스를 현재 위치로 초기화
        current_local_idx = start_idx 
        
        while True:
            # 입력 윈도우: [current_local_idx : current_local_idx + input_len]
            # 예측 시점: current_local_idx + input_len

            # 현재 윈도우의 예측 시점 인덱스 = 시작 인덱스 + input_len
            pred_timestamp_idx = current_local_idx + input_len
            
            if pred_timestamp_idx >= len(df):
                break
            
            # 예측 시점의 실제 타임스탬프 
            pred_time = df.loc[pred_timestamp_idx, "date_time"]

            # 시각화 종료 시점(view_end) 넘으면 중단
            if pred_time > view_end:
                break
            
            # 글로벌 인덱스는 입력 윈도우의 시작점 기준
            # 전체 df_all 기준에서의 현재 샘플의 전역 인덱스 
            # global_idx example: station_01 (0~999), station_02 (1000~1999), ...
            global_idx = df.loc[current_local_idx, 'global_idx']
            # 예측 정보 저장
            #    (글로벌 인덱스, 입력 윈도우 시작 로컬 인덱스, 예측 시점 로컬 인덱스)
            prediction_info.append((global_idx, current_local_idx, pred_timestamp_idx))
            
            current_local_idx += 1 # 다음 윈도우로 이동
            
            if len(prediction_info) > 10000:
                print(f"[경고] Station {sid}: 예측 인덱스 과다")
                break

        if not prediction_info:
            print(f"[경고] Station {sid}: 예측 가능한 인덱스 없음")
            continue

        print(f"예측 샘플 수: {len(prediction_info)}")
        print(f"글로벌 인덱스 범위: {prediction_info[0][0]} ~ {prediction_info[-1][0]}")
        print(f"예측 시점 범위: {df.loc[prediction_info[0][2], 'date_time']} ~ {df.loc[prediction_info[-1][2], 'date_time']}")

        # ========== 모델 예측 수행 ==========
        model.eval()
        pred_map = {}
        batch_offset = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                xb, s_idx, meta = _unpack_batch(batch)
                batch_size = xb.size(0)
                batch_start = batch_offset
                batch_end = batch_offset + batch_size

                # 현재 배치에 포함된 예측 샘플 찾기
                samples_in_batch = [(g, inp_start, pred_ts) 
                                   for g, inp_start, pred_ts in prediction_info 
                                   if batch_start <= g < batch_end]
                
                if samples_in_batch:
                    xb_dev = xb.to(device, non_blocking=True)
                    
                    if permute_to_BFL and xb_dev.dim() == 3:
                        xb_dev = xb_dev.permute(0, 2, 1)

                    kwargs = {} # station_idx, meta 전달용
                    if s_idx is not None:
                        kwargs["station_idx"] = s_idx.to(device, non_blocking=True)
                    if meta is not None:
                        meta_dev = meta.to(device, non_blocking=True)
                        if meta_dev.dim() == 3:
                            meta_dev = meta_dev.mean(dim=1)
                        kwargs["meta"] = meta_dev

                    # 모델 forward
                    try:
                        yhat = model(xb_dev, **kwargs)
                    except TypeError:
                        try:
                            yhat = model(xb_dev, 
                                       kwargs.get("station_idx", None), 
                                       kwargs.get("meta", None))
                        except TypeError:
                            yhat = model(xb_dev)

                    yhat_np = yhat.detach().cpu().numpy()  # [B, output_len]
                    
                    for g_idx, inp_start, pred_ts_idx in samples_in_batch:
                        batch_offset_idx = g_idx - batch_start
                        pred_seq = _inv_standard_log1p(yhat_np[batch_offset_idx].reshape(-1))
                        
                        # ✅ 핵심 수정: pred_ts_idx 사용 (예측 시점의 실제 타임스탬프)
                        pred_timestamp = df.loc[pred_ts_idx, "date_time"]
                        
                        # output_len만큼 매핑 (일반적으로 1개)
                        for step in range(min(output_len, len(pred_seq))):
                            if pred_ts_idx + step < len(df):
                                ts = df.loc[pred_ts_idx + step, "date_time"]
                                ts = pd.Timestamp(ts)
                                if ts <= view_end:
                                    pred_map[ts] = float(pred_seq[step])
                    
                    # 첫 배치 디버깅
                    if batch_idx == 0 and samples_in_batch:
                        g0, inp0, pred0 = samples_in_batch[0]
                        offset0 = g0 - batch_start
                        print(f"\n[디버깅 - 첫 예측 샘플]")
                        print(f"  입력 윈도우 로컬 인덱스: {inp0} ~ {inp0 + input_len - 1}")
                        print(f"  예측 타임스탬프 로컬 인덱스: {pred0}")
                        print(f"  예측 시점: {df.loc[pred0, 'date_time']}")
                        print(f"  배치 내 오프셋: {offset0}")
                        if s_idx is not None:
                            print(f"  station_idx: {s_idx[offset0].item()}")
                        print(f"  모델 출력 (표준화됨): {yhat_np[offset0][:3]}")
                        print(f"  역변환 후 (kW): {_inv_standard_log1p(yhat_np[offset0].reshape(-1))[:3]}")

                batch_offset += batch_size
                
                if batch_offset > max([g for g, _, _ in prediction_info]):
                    break

        if not pred_map:
            print(f"[경고] Station {sid}: 예측 결과 생성 실패")
            continue

        print(f"생성된 예측 포인트 수: {len(pred_map)}")

        # ========== 시각화 ==========
        pred_series = pd.Series(pred_map).sort_index()
        
        plt.figure(figsize=(14, 6))
        
        plt.plot(input_part["date_time"], input_part["power"], 
                color='#1f77b4', label="Input (Actual)", linewidth=2.5, marker='o', markersize=5)
        
        plt.plot(output_part["date_time"], output_part["power"], 
                color='#ff7f0e', label="Output (Actual)", linewidth=2.5, marker='s', markersize=4)
        
        plt.plot(pred_series.index, pred_series.values, 
                color='#2ca02c', linestyle="--", label="Predicted", 
                linewidth=2.5, marker="^", markersize=5, alpha=0.85)
    
        
        title = f"PVPlant Power Forecast | In_window={input_len}, Out_window={output_len}"
        if sid is not None:
            title += f" | Station {sid}"
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Power (kW)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc="best", fontsize=11)
        plt.xlim([input_start, view_end])
        plt.tight_layout()

        if save_path is not None:
            if len(station_list) == 1:
                out_path = save_path
            else:
                base, ext = (save_path.rsplit(".", 1) + ["png"])[:2]
                out_path = f"{base}_{sid}.{ext}"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[저장] {out_path}")
        
        plt.show()

"""
Owaolabi 예측 결과 시각화 함수
"""
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