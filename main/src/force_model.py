"""하중 예측 GBDT 모델 (단조 제약 포함). optimize.py와 Streamlit 앱에 바로 끼울 수 있게 만들었다.

원본: ojs/force_model.py (오정수). 개발 과정은 ojs/01~05 노트북.

입력은 pjh 열 이름 그대로의 스탠드 단위 표다 (pjh/data_utils.py의 load_clean() + add_features()와 같은 열).
    Grade, Stand ID, Entry gauge (mm), Exit gauge (mm), Radius (mm), Width (mm), Temperature (K)

사용 예
    import sys; sys.path.insert(0, "main/src")
    from force_model import train, make_predict_fn
    from optimize import load_data, limits_from_data, coil_from_data, optimize
    df = load_data()
    model = train(df)                      # 약 30초
    predict_fn = make_predict_fn(model)    # rows(DataFrame) -> 하중(MN) 배열
    res = optimize(coil_from_data(2744, df), predict_fn, limits_from_data(df))

설정 근거
    - 학습률 0.1, 나무 1,400개: ojs/04_gbdt_h1h2.ipynb (H1·H2 가설 시험)
    - 단조 제약: 같은 스탠드에서 더 누르면(출측 두께↓, 압하율↑) 하중이 줄지 않게 함.
      제약이 없으면 5~6% 행에서 "더 눌렀는데 하중 감소"가 나와 최적화가 그 구간을 악용함 (ojs/02_ai_models.ipynb 8번).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

V1, HO1 = 1096.0, 27.7      # 질량 보존 속도 계산용 (논문 가정값, ojs/01_physics_baseline.ipynb)
GRADES = [f"Grade{i:02d}" for i in range(1, 11)]

# 입력 변수와 단조 방향 (+1: 커지면 하중↑, -1: 커지면 하중↓, 0: 제약 없음)
FEATURES = {
    "grade": 0,          # 강종 (범주형, 제약 불가)
    "stand_no": 0,
    "h_in": 0,
    "h_out": -1,         # 출측이 얇을수록 = 더 누를수록 하중↑
    "W": 0,
    "R": 0,
    "T_K": 0,
    "reduction": 1,
    "dh": 1,
    "L": 1,
    "strain": 1,
    "strain_rate": 1,
}

PARAMS = dict(learning_rate=0.1, max_iter=1400, max_leaf_nodes=31, min_samples_leaf=40,
              l2_regularization=1.0, early_stopping=False, random_state=0)


def features(rows: pd.DataFrame) -> pd.DataFrame:
    """pjh 열 이름의 표 → 모델 입력 표."""
    h_in = rows["Entry gauge (mm)"].to_numpy(float)
    h_out = rows["Exit gauge (mm)"].to_numpy(float)
    R = rows["Radius (mm)"].to_numpy(float)
    dh = h_in - h_out
    L = np.sqrt(R * dh)
    strain = np.log(h_in / h_out)
    X = pd.DataFrame({
        "grade": pd.Categorical(rows["Grade"].to_numpy(), categories=GRADES),
        "stand_no": rows["Stand ID"].str[1].astype(int).to_numpy(),
        "h_in": h_in,
        "h_out": h_out,
        "W": rows["Width (mm)"].to_numpy(float),
        "R": R,
        "T_K": rows["Temperature (K)"].to_numpy(float),
        "reduction": dh / h_in,
        "dh": dh,
        "L": L,
        "strain": strain,
        "strain_rate": strain * (V1 * HO1 / h_out) / L,
    })
    return X[list(FEATURES)]


def train(df: pd.DataFrame, monotonic: bool = True, **changes) -> HistGradientBoostingRegressor:
    """pjh 정리 데이터(load_clean + add_features)로 학습. 목표는 하중(MN)."""
    params = {**PARAMS, **changes}
    if monotonic:
        params["monotonic_cst"] = list(FEATURES.values())
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", **params)
    return model.fit(features(df), df["Force (N)"].to_numpy() / 1e6)


def make_predict_fn(model):
    """khj/optimize.py의 predict_fn 형식: rows(DataFrame) -> 하중(MN) 배열."""
    def predict_fn(rows):
        return np.clip(model.predict(features(rows)), 0, None)
    return predict_fn
