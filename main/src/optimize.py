"""압하 배분 최적화.

원본: khj/optimize.py (김하진). main으로 옮기면서 import 경로를 바꾸고, 맨 아래 실행 예시는 main/main.py로 옮겼다.
최종 모델은 force_model.py의 단조 제약 GBDT를 predict_fn으로 끼워 쓴다.

 어떤 하중 예측 모델이 와도 그대로 쓸 수 있게 만들었다.

핵심 아이디어
    최적화는 모델 내부를 몰라도 된다. "배분을 넣으면 스탠드별 하중이 나오는 함수"만 있으면 된다.
    그래서 모델은 predict_fn 하나로 받는다.

        predict_fn(rows: DataFrame) -> np.ndarray   # 행마다 하중(MN)

    rows는 pjh/data_utils.py의 load_clean() + add_features()와 같은 컬럼을 가진 스탠드 단위 표다.
    그 데이터로 학습한 모델이면 GBDT, MLP, 물리식 무엇이든 그대로 끼울 수 있다.

최적화 방식 (무작위 탐색)
    1. 스탠드별 압하율 후보를 수천 개 만든다. 과거 운영 범위 안에서, 최종 두께가 목표와 같도록 맞춘다.
    2. 모델로 후보 전부의 F1~F5 하중을 한 번에 예측한다.
    3. 점수(기본: 최대 이용률 + 0.5 × 이용률 편차)가 가장 낮은 후보를 고른다.

사용 예
    from optimize import load_data, limits_from_data, coil_from_data, optimize, physics_predictor
    df = load_data()
    lim = limits_from_data(df)
    coil = coil_from_data(2744, df)
    result = optimize(coil, physics_predictor(df), lim)

실행: python main/main.py
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import FORCE, STANDS, add_features, load_clean  # 같은 폴더 (main/src)

N_STANDS = len(STANDS)


# ---------------------------------------------------------------------------
# 1. 입력 형식
# ---------------------------------------------------------------------------
@dataclass
class Coil:
    """최적화할 코일의 고정 조건. 압하율만 바꾸고 나머지는 그대로 둔다."""
    grade: str                  # "Grade01" ~ "Grade10"
    caster: str                 # "S" 또는 "N"
    width: float                # 판 폭 (mm)
    h0: float                   # F1 입측 두께 (mm)
    h_final: float              # 목표 최종 두께 = F5 출측 (mm)
    radii: list                 # F1~F5 작업롤 반경 (mm), 5개
    temps_K: list = None        # F1~F5 소재 온도 (K). 없으면 스탠드별 과거 중앙값을 쓴다
    reductions: list = None     # 현재 압하율 F1~F5 (있으면 추천과 비교한다)


@dataclass
class Limits:
    """데이터에서 정한 운영 범위와 이용률 기준."""
    red_lo: np.ndarray          # 스탠드별 압하율 하한
    red_hi: np.ndarray          # 스탠드별 압하율 상한
    cap: np.ndarray             # 스탠드별 이용률 기준 하중 (MN). 이용률 = 하중 / cap
    temp_K: np.ndarray          # 스탠드별 온도 중앙값 (K), 온도를 모를 때 쓴다
    grades: list                # 선택 가능한 강종


def load_data() -> pd.DataFrame:
    """정리된 데이터 + 파생 변수 (pjh 기준과 동일)."""
    df, _ = load_clean()
    return add_features(df)


def limits_from_data(df, lo_q=0.01, hi_q=0.99, cap_q=0.995) -> Limits:
    """과거 운영 범위(압하율 1~99%)와 이용률 기준(하중 상위 0.5%)을 데이터에서 계산한다."""
    g = df.groupby("Stand ID")
    return Limits(
        red_lo=g["reduction"].quantile(lo_q).reindex(STANDS).to_numpy(),
        red_hi=g["reduction"].quantile(hi_q).reindex(STANDS).to_numpy(),
        cap=g["force_MN"].quantile(cap_q).reindex(STANDS).to_numpy(),
        temp_K=g["Temperature (K)"].median().reindex(STANDS).to_numpy(),
        grades=sorted(df["Grade"].unique()),
    )


def coil_from_data(coil_id, df) -> Coil:
    """실제 코일 하나의 조건을 Coil로 만든다 (시연용 '샘플 코일 불러오기')."""
    s = df[df["Coil ID"] == coil_id].sort_values("Stand ID")
    if len(s) != N_STANDS:
        raise ValueError(f"코일 {coil_id}: 스탠드 {len(s)}개 (정리된 데이터에 없거나 불완전)")
    return Coil(
        grade=s["Grade"].iloc[0],
        caster=s["Caster ID"].iloc[0],
        width=float(s["Width (mm)"].iloc[0]),
        h0=float(s["Entry gauge (mm)"].iloc[0]),
        h_final=float(s["Exit gauge (mm)"].iloc[-1]),
        radii=s["Radius (mm)"].tolist(),
        temps_K=s["Temperature (K)"].tolist(),
        reductions=s["reduction"].tolist(),
    )


# ---------------------------------------------------------------------------
# 2. 배분 → 모델 입력 표
# ---------------------------------------------------------------------------
def schedules_to_rows(coil: Coil, R: np.ndarray, lim: Limits) -> pd.DataFrame:
    """압하율 배분 R (후보 수 × 5)을 스탠드 단위 표 (후보 수 × 5 행)로 바꾼다.
    컬럼은 원본 CSV + pjh add_features()와 같아서, 그 데이터로 학습한 모델에 바로 넣을 수 있다."""
    R = np.atleast_2d(R)
    n = len(R)
    h = coil.h0 * np.c_[np.ones(n), np.cumprod(1 - R, axis=1)]   # 각 스탠드 입·출측 두께
    temps = coil.temps_K if coil.temps_K is not None else lim.temp_K
    rows = pd.DataFrame({
        "Grade": coil.grade,
        "Coil ID": -1,
        "Caster ID": coil.caster,
        "Stand ID": np.tile(STANDS, n),
        "Entry gauge (mm)": h[:, :-1].ravel(),
        "Exit gauge (mm)": h[:, 1:].ravel(),
        "Radius (mm)": np.tile(coil.radii, n),
        "Width (mm)": coil.width,
        "Temperature (K)": np.tile(temps, n),
        FORCE: np.nan,                                          # 예측할 값이라 비워 둔다
    })
    return add_features(rows)


# ---------------------------------------------------------------------------
# 3. 후보 만들기, 점수, 최적화
# ---------------------------------------------------------------------------
def sample_candidates(coil: Coil, lim: Limits, n=3000, rng=None) -> np.ndarray:
    """과거 운영 범위 안에서, 최종 두께가 목표와 정확히 같은 압하율 후보를 만든다.

    최종 두께 = h0 × (1-r1)(1-r2)…(1-r5) 이므로, log(1-r)의 합이 log(h_final/h0)가 되도록
    무작위 후보의 비율을 맞춘다. 맞춘 뒤 범위를 벗어난 후보는 버린다."""
    rng = np.random.default_rng(rng)
    target = np.log(coil.h_final / coil.h0)
    R = rng.uniform(lim.red_lo, lim.red_hi, size=(n, N_STANDS))
    logs = np.log(1 - R)
    logs *= (target / logs.sum(axis=1))[:, None]
    R = 1 - np.exp(logs)
    return R[((R >= lim.red_lo) & (R <= lim.red_hi)).all(axis=1)]


def balance_objective(F, lim: Limits) -> np.ndarray:
    """기본 점수: 최대 이용률 + 0.5 × 스탠드 간 이용률 편차. 낮을수록 좋다."""
    u = F / lim.cap
    return u.max(axis=1) + 0.5 * u.std(axis=1)


def predict_schedules(coil, R, predict_fn, lim) -> np.ndarray:
    """배분 여러 개의 F1~F5 하중 (후보 수 × 5, MN)."""
    rows = schedules_to_rows(coil, R, lim)
    return np.asarray(predict_fn(rows), dtype=float).reshape(-1, N_STANDS)


def in_range(R, lim) -> bool:
    return bool(((np.asarray(R) >= lim.red_lo) & (np.asarray(R) <= lim.red_hi)).all())


def optimize(coil: Coil, predict_fn, lim: Limits, n=3000, objective=balance_objective, seed=0) -> dict:
    """가장 점수가 낮은 압하 배분을 찾는다. 현재 배분(coil.reductions)이 있으면 같이 평가해 비교한다."""
    if coil.grade not in lim.grades:
        raise ValueError(f"{coil.grade}: 학습 데이터에 없는 강종이라 예측을 믿을 수 없어요")
    R = sample_candidates(coil, lim, n, seed)
    if len(R) == 0:
        raise ValueError("범위 안에서 최종 두께를 맞추는 후보가 없어요 (두께 조건이 과거 범위 밖)")

    F = predict_schedules(coil, R, predict_fn, lim)
    score = objective(F, lim)
    b = int(np.argmin(score))
    result = {
        "best": _summary(R[b], F[b], score[b], lim),
        "n_candidates": len(R),
    }
    if coil.reductions is not None:
        cur = np.asarray(coil.reductions)
        Fc = predict_schedules(coil, cur, predict_fn, lim)
        result["current"] = _summary(cur, Fc[0], objective(Fc, lim)[0], lim)
        result["current"]["in_range"] = in_range(cur, lim)
        u0, u1 = result["current"]["util"].max(), result["best"]["util"].max()
        result["util_max_drop_pct"] = (1 - u1 / u0) * 100
    return result


def _summary(r, f, score, lim) -> dict:
    u = f / lim.cap
    return {
        "reductions": np.round(r, 4),
        "forces_MN": np.round(f, 2),
        "util": np.round(u, 3),
        "bottleneck": STANDS[int(np.argmax(u))],
        "score": round(float(score), 4),
    }


# ---------------------------------------------------------------------------
# 4. 모델 끼우기 (어댑터). 새 모델은 여기에 함수 하나만 추가하면 된다.
# ---------------------------------------------------------------------------
def sklearn_predictor(model, prepare, unit="MN"):
    """scikit-learn처럼 .predict()가 있는 모델을 predict_fn으로 감싼다.
    prepare(rows) -> 모델 입력 X. 학습할 때 쓴 전처리와 똑같아야 한다.
    PyTorch MLP도 같은 모양의 함수(rows -> MN 배열)로 감싸면 그대로 쓸 수 있다."""
    scale = 1e-6 if unit == "N" else 1.0

    def predict_fn(rows):
        return np.asarray(model.predict(prepare(rows))) * scale
    return predict_fn


def physics_predictor(df):
    """논문 M3 형태의 물리식: log F = a + b·log(변형 속도) + log(폭 × 접촉 길이).
    변형 속도는 질량 보존으로 추정한다 (속도 ∝ 1/출측 두께). 계수 a, b는 df로 맞춘다."""
    def design(d):
        edot = d["strain"] / d["Exit gauge (mm)"] / d["contact_len"]
        return np.log(edot.to_numpy()), np.log((d["Width (mm)"] * d["contact_len"]).to_numpy() / 1e6)

    x, offset = design(df)
    A = np.c_[np.ones_like(x), x]
    a, b = np.linalg.lstsq(A, np.log(df["force_MN"].to_numpy()) - offset, rcond=None)[0]

    def predict_fn(rows):
        x, offset = design(rows)
        return np.exp(a + b * x + offset)
    return predict_fn


GBDT_FEATURES = ["Entry gauge (mm)", "Exit gauge (mm)", "Radius (mm)", "Width (mm)",
                 "Temperature (K)", "draft", "reduction", "strain", "contact_len"]


def gbdt_prepare(rows):
    """GBDT 예시용 전처리: 숫자 변수 + 스탠드 번호 + 강종 번호 + 연주기."""
    X = rows[GBDT_FEATURES].copy()
    X["stand_no"] = rows["Stand ID"].str[1].astype(int)
    X["grade_no"] = rows["Grade"].str[-2:].astype(int)
    X["caster_S"] = (rows["Caster ID"] == "S").astype(int)
    return X


def train_example_gbdt(df):
    """예시 GBDT. 실제 모델이 나오면 이 함수 대신 그 모델을 sklearn_predictor로 감싸면 된다."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = gbdt_prepare(df)
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0,
                                          categorical_features=[X.columns.get_loc("grade_no")])
    return model.fit(X, df["force_MN"])
