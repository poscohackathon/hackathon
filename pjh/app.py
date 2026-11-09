"""STEEL FLOW AI - AI 기반 열연 사상압연(F1~F5) 시뮬레이션 및 압하 배분 최적화

실행 (pjh 폴더에서):
    pip install -r requirements.txt
    streamlit run app.py

[주의] 아래 세 가지는 '데모/가정'이며 화면에도 그렇게 표시된다.
  1) 예측 모델   : GBDT (단조 제약, ojs/force_model.py). 앱 시작 때 한 번 학습 (약 20초)  -> 3. 예측 모델 구간
  2) 경고 기준   : 이용률 = 하중 / 스탠드별 과거 최대 하중(상위 0.5%). 주의 0.9, 위험 1.0 (실제 설비 정격 아님) -> 1. 설정 구간
  3) 최적화      : khj/optimize.py의 optimize() (과거 운영 범위 안 무작위 탐색)  -> 4. 최적화 구간

코드 구성
  1. 설정(CONFIG)      : 색상, 이용률 기준, 파일 경로 등 바꾸기 쉬운 값
  2. 데이터 준비       : CSV 읽기, 연속성 깨진 코일 제외, 관측 범위 계산
  3. 예측 모델         : predict_batch()만 바꾸면 최종 모델로 교체 가능
  4. 압하 배분 최적화  : khj/optimize.py 호출
  5. 경고 판정         : 정상 / 주의 / 위험
  6. AI Agent          : 규칙 기반 공정 분석 (LLM 연결 자리 포함)
  7. 시각화            : 공정 애니메이션, 그래프, 카드
  8. 화면(main)        : Streamlit UI와 session_state
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ojs"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "khj"))
from force_model import train as train_force_model, make_predict_fn  # noqa: E402
import optimize as khj_opt  # noqa: E402  (김하진 최적화 코드)

# =====================================================================================
# 1. 설정 (CONFIG)
# =====================================================================================
DATA_CSV = Path(__file__).resolve().parent.parent / "main" / "raw_data" / "metals.csv"
STANDS = ["F1", "F2", "F3", "F4", "F5"]

# ---- 경고 기준: 이용률 (실제 설비 정격이 아니다!) ---------------------------------------
#   이용률 = 예측 하중 / 그 스탠드의 과거 최대 하중 (데이터의 상위 0.5% 값, load_resources에서 계산)
#   주의: 이용률 0.9 이상, 위험: 이용률 1.0 이상
CAUTION_UTIL, DANGER_UTIL = 0.9, 1.0
#   LIMITS: 스탠드 -> (주의 하중 MN, 위험 하중 MN). 아래는 초기값이며, 실행할 때 apply_caps()가 데이터에서 다시 채운다.
_CAP0 = {"F1": 36.43, "F2": 33.68, "F3": 23.62, "F4": 19.45, "F5": 14.31}
LIMITS = {s: (CAUTION_UTIL * c, DANGER_UTIL * c) for s, c in _CAP0.items()}


def apply_caps(res: dict):
    """데이터에서 계산한 스탠드별 과거 최대 하중으로 경고 기준(LIMITS)을 갱신한다. (스크립트가 다시 실행될 때마다 호출)"""
    for s, cap in zip(STANDS, res["caps"]):
        LIMITS[s] = (CAUTION_UTIL * float(cap), DANGER_UTIL * float(cap))


def caps_arr() -> np.ndarray:
    """스탠드별 과거 최대 하중 (이용률 1.0 에 해당하는 하중, MN)."""
    return np.array([LIMITS[s][1] / DANGER_UTIL for s in STANDS])


CONCENTRATION_ALERT_SHARE = 0.30   # 한 스탠드가 전체 하중의 30%를 넘으면 '하중 집중' 경고 (데모용)

# ---- 샘플 코일 (김하진 인계 문서의 시연 코일. 데이터에서 실제 조건과 실측 하중을 불러온다) --------
SAMPLE_COILS = {2744: "메인 시연 · F1 병목", 2599: "보조 시연 · F1 한계 직전", 2086: "후보 · Grade01",
                6175: "후보 · Grade06", 6920: "후보 · Grade08"}

# ---- 색상 (다크 테마) ------------------------------------------------------------------
C_BG, C_CARD, C_BORDER = "#0c0c0f", "#17171c", "#2b2b33"
C_TEXT, C_MUTED = "#ececf0", "#9b9ba6"
C_ACCENT, C_STEEL = "#ff8f2e", "#6f86a8"   # 포인트색(주황), 보조색(강철 블루)
C_OK, C_WARN, C_DANGER = "#22c55e", "#f5c542", "#ef4444"
STATUS = {
    "ok": ("정상", C_OK),
    "warn": ("주의", C_WARN),
    "danger": ("위험", C_DANGER),
}

# =====================================================================================
# 2. 데이터 준비
# =====================================================================================
def drop_broken_coils(raw: pd.DataFrame) -> pd.DataFrame:
    """연속성이 깨진 코일(스탠드 구성 불완전, 두께 불연속, 폭 변동)을 코일 단위로 제외한다."""
    d = raw.sort_values(["Coil ID", "Stand ID"]).reset_index(drop=True)
    g = d.groupby("Coil ID")
    complete = (g["Stand ID"].size() == 5) & (g["Stand ID"].nunique() == 5)
    gap = (d["Exit gauge (mm)"] - g["Entry gauge (mm)"].shift(-1)).abs()
    gap_bad = set(d.loc[gap > 0.01, "Coil ID"])
    wr = g["Width (mm)"].max() - g["Width (mm)"].min()
    width_bad = set(wr.index[wr > 0.01])
    broken = set(complete.index[~complete]) | gap_bad | width_bad
    return d[~d["Coil ID"].isin(broken)].reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["draft"] = d["Entry gauge (mm)"] - d["Exit gauge (mm)"]
    d["reduction"] = d["draft"] / d["Entry gauge (mm)"]
    d["strain"] = np.log(d["Entry gauge (mm)"] / d["Exit gauge (mm)"])
    d["force_MN"] = d["Force (N)"] / 1e6
    return d


@st.cache_resource(show_spinner="학습 데이터와 GBDT 예측 모델을 준비하는 중입니다 (처음 한 번, 약 20초)...")
def load_resources() -> dict:
    """데이터 읽기, 관측 범위 계산, GBDT 예측 모델 학습 (앱 실행 중 한 번만 수행)."""
    if not DATA_CSV.exists():
        return {"error": f"데이터 파일을 찾을 수 없습니다: {DATA_CSV}"}
    raw = pd.read_csv(DATA_CSV, usecols=range(10))   # CSV 뒤쪽 빈 컬럼 18개는 버린다
    d = add_features(drop_broken_coils(raw))

    grades = sorted(d["Grade"].unique())
    model = make_predict_fn(train_force_model(d))   # 최종 모델 (ojs/force_model.py), 앱 시작 때 한 번

    q = d.groupby("Stand ID")["reduction"].quantile([0.01, 0.5, 0.99]).unstack().loc[STANDS]
    first, last = d[d["Stand ID"] == "F1"], d[d["Stand ID"] == "F5"]

    # 현행 압하 배분 규칙: 코일 전체 변형률 중 스탠드별 몫의 중앙값
    tot = d.groupby("Coil ID")["strain"].transform("sum")
    share = (d["strain"] / tot).groupby(d["Stand ID"]).median().loc[STANDS].to_numpy()
    share = share / share.sum()

    # 대표 코일: 입구 두께, 최종 두께, 폭이 중앙값에 가장 가까운 코일
    c = d.groupby("Coil ID").agg(h0=("Entry gauge (mm)", "first"), w=("Width (mm)", "first"),
                                 hf=("Exit gauge (mm)", "last"))
    rep_id = (((c - c.median()) / c.std()) ** 2).sum(axis=1).idxmin()
    rep = d[d["Coil ID"] == rep_id].sort_values("Stand ID")

    temp_med = d.groupby("Stand ID")["Temperature (K)"].median().loc[STANDS].to_numpy()
    cv_coil = d.groupby("Coil ID")["force_MN"].agg(lambda x: x.std(ddof=0) / x.mean())

    # 입력 슬라이더 범위: 데이터의 최소~최대 (과거 운영 범위 1~99%는 red_lo/red_hi로 따로 두고 신뢰도 판정에 쓴다)
    fl, ce = (lambda x, k: math.floor(x * k) / k), (lambda x, k: math.ceil(x * k) / k)
    red_min = d.groupby("Stand ID")["reduction"].min().loc[STANDS].to_numpy()
    red_max = d.groupby("Stand ID")["reduction"].max().loc[STANDS].to_numpy()

    # 샘플 코일: 정리된 데이터는 코일마다 F1~F5 5행이고 (코일, 스탠드) 순으로 정렬돼 있다
    n5 = len(d) // 5
    col5 = lambda name: d[name].to_numpy().reshape(n5, 5)
    ids5, pos = col5("Coil ID")[:, 0], {}
    pos = {int(c): i for i, c in enumerate(ids5)}
    f5, red5, ent5, ex5 = col5("force_MN"), col5("reduction"), col5("Entry gauge (mm)"), col5("Exit gauge (mm)")
    wid5, tmp5, gr5, ca5 = col5("Width (mm)"), col5("Temperature (K)"), col5("Grade"), col5("Caster ID")
    coils = {}
    for cid in SAMPLE_COILS:
        if cid in pos:
            k = pos[cid]
            coils[cid] = {"grade": str(gr5[k, 0]), "caster": str(ca5[k, 0]), "width": float(wid5[k, 0]),
                          "entry": float(ent5[k, 0]), "final": float(ex5[k, 4]), "temp_c": float(tmp5[k, 0]) - 273.15,
                          "reds": red5[k].astype(float), "forces": f5[k].astype(float),
                          "radii": col5("Radius (mm)")[k].astype(float), "temps": tmp5[k].astype(float)}

    return {
        "coils": coils, "red_min": np.floor(red_min * 200) / 200, "red_max": np.ceil(red_max * 200) / 200,
        "model": model, "grades": grades, "n_grades": len(grades),
        "grade_counts": d.groupby("Grade")["Coil ID"].nunique().to_dict(),
        "red_lo": q[0.01].to_numpy(), "red_med": q[0.5].to_numpy(), "red_hi": q[0.99].to_numpy(),
        "strain_share": share,
        "entry_range": (fl(first["Entry gauge (mm)"].min(), 10), ce(first["Entry gauge (mm)"].max(), 10)),
        "final_range": (fl(last["Exit gauge (mm)"].min(), 20), ce(last["Exit gauge (mm)"].max(), 20)),
        "width_range": (math.floor(d["Width (mm)"].min()), math.ceil(d["Width (mm)"].max())),
        "width_med": float(d["Width (mm)"].median()),
        "temp_med": temp_med,                                   # 스탠드별 중앙값 (K)
        "temp1_range_c": (math.floor(first["Temperature (K)"].min() - 273.15), math.ceil(first["Temperature (K)"].max() - 273.15)),
        "radius_med": d.groupby("Stand ID")["Radius (mm)"].median().loc[STANDS].to_numpy(),
        "cv_med": float(cv_coil.median()),
        "caps": d.groupby("Stand ID")["force_MN"].quantile(0.995).loc[STANDS].to_numpy(),   # 스탠드별 과거 최대 하중(상위 0.5%)
        "rep": {"grade": str(rep["Grade"].iloc[0]), "width": float(rep["Width (mm)"].iloc[0]),
                "entry": float(rep["Entry gauge (mm)"].iloc[0]), "final": float(rep["Exit gauge (mm)"].iloc[-1]),
                "temp_c": float(rep["Temperature (K)"].iloc[0]) - 273.15},
        "n_rows": len(d), "n_coils": int(d["Coil ID"].nunique()),
    }


# =====================================================================================
# 3. 예측 모델  ->  최종 모델이 정해지면 predict_batch()만 교체하면 된다
# =====================================================================================
MODEL_NAME = "GBDT (단조 제약)"
MODEL_IS_PLACEHOLDER = False


def stand_temps(res: dict, temp1_c: float) -> np.ndarray:
    """F1 온도를 입력받아 스탠드별 온도(K)를 만든다 (스탠드 간 온도 차는 과거 중앙값 차이를 사용)."""
    return temp1_c + 273.15 + (res["temp_med"] - res["temp_med"][0])


def stand_conditions(res: dict, inp: dict, smp: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """스탠드별 롤 반경(mm)과 온도(K).
    샘플 코일을 불러왔으면 그 코일의 실제 값을 쓰고 (F1 온도를 바꾸면 같은 만큼 평행 이동), 아니면 과거 중앙값을 쓴다."""
    if smp:
        temps = np.asarray(smp["temps"]) + (inp["temp_c"] - smp["inputs"]["temp_c"])
        return np.asarray(smp["radii"]), temps
    return res["radius_med"], stand_temps(res, inp["temp_c"])


def predict_batch(res: dict, grade: str, width: float, entry: float, exits: np.ndarray, temps: np.ndarray,
                  radii: np.ndarray) -> np.ndarray:
    """여러 압하 배분 후보의 스탠드별 하중(MN)을 한 번에 예측한다.

    Args:
        exits: (후보 수, 5) 스탠드별 출구 두께 (mm)
        temps: (5,) 스탠드별 온도 (K)
        radii: (5,) 스탠드별 롤 반경 (mm)
    Returns:
        (후보 수, 5) 예측 하중 (MN)

    [교체 방법] 최종 AI 모델을 쓸 때는 이 함수 안의 계산만 바꾸고 입출력 형식은 유지한다.
    """
    n = exits.shape[0]
    entries = np.hstack([np.full((n, 1), entry), exits[:, :-1]])
    rows = pd.DataFrame({"Grade": grade, "Stand ID": np.tile(STANDS, n),
                         "Entry gauge (mm)": entries.ravel(), "Exit gauge (mm)": exits.ravel(),
                         "Radius (mm)": np.tile(radii, n), "Width (mm)": width,
                         "Temperature (K)": np.tile(temps, n)})
    return res["model"](rows).reshape(n, 5)


# =====================================================================================
# 4. 압하 배분 최적화 (khj/optimize.py)
# =====================================================================================
def exits_from_reductions(entry: float, reds: np.ndarray) -> np.ndarray:
    """압하율 (…, 5) -> 스탠드별 출구 두께 (…, 5)."""
    return entry * np.cumprod(1.0 - reds, axis=-1)


def default_reductions(res: dict, entry: float, final: float) -> np.ndarray:
    """현행 배분: 총 변형률을 과거 코일들의 스탠드별 몫(중앙값)대로 나눈다."""
    eps = res["strain_share"] * math.log(entry / final)
    return 1.0 - np.exp(-eps)


def optimize_schedule(res: dict, inp: dict, cur_reds: np.ndarray) -> dict:
    """김하진 khj/optimize.py의 optimize()로 압하 배분을 추천한다 (발표 숫자와 같은 알고리즘).

    제약조건: 최종 두께 = 목표값, 모든 스탠드 압하율은 과거 운영 범위(1~99%) 안
    목적: 최대 이용률 + 0.5 x 스탠드 간 이용률 표준편차 최소화 (이용률 = 하중 / 스탠드별 과거 최대 하중)
    방식: 범위 안 무작위 후보 3,000개를 GBDT로 한 번에 예측해 점수가 가장 낮은 것을 고른다 (seed 0)
    """
    coil = khj_opt.Coil(grade=inp["grade"], caster=inp["caster"], width=inp["width"], h0=inp["entry"],
                        h_final=inp["final"], radii=list(inp["radii"]), temps_K=list(inp["temps"]),
                        reductions=list(cur_reds))
    lim = khj_opt.Limits(red_lo=res["red_lo"], red_hi=res["red_hi"], cap=caps_arr(), temp_K=res["temp_med"],
                         grades=res["grades"])
    try:
        out = khj_opt.optimize(coil, res["model"], lim)
    except ValueError as e:
        return {"error": str(e)}
    best = np.asarray(out["best"]["reductions"], dtype=float)
    return {"reductions": best, "n_evaluated": out["n_candidates"],
            "feasible": bool(out["best"]["util"].max() < DANGER_UTIL),
            "changed": bool(np.abs(best - cur_reds).max() > 1e-4)}


# =====================================================================================
# 5. 경고 판정
# =====================================================================================
def classify(force: float, stand: str) -> str:
    caution, danger = LIMITS[stand]
    if force >= danger:
        return "danger"
    if force >= caution:
        return "warn"
    return "ok"


def build_result(res: dict, inp: dict, reds: np.ndarray, label: str) -> dict:
    """압하율(5개)로부터 출구 두께, 예측 하중, 경고 상태, KPI를 계산한다."""
    entry = inp["entry"]
    exits = exits_from_reductions(entry, reds)
    entries = np.concatenate([[entry], exits[:-1]])
    temps, radii = inp["temps"], inp["radii"]
    forces = predict_batch(res, inp["grade"], inp["width"], entry, exits[None, :], temps, radii)[0]
    drafts = entries - exits
    status = [classify(f, s) for f, s in zip(forces, STANDS)]
    in_range = [bool(res["red_lo"][i] <= reds[i] <= res["red_hi"][i]) for i in range(5)]
    shares = forces / forces.sum()
    top = int(np.argmax(forces))
    return {
        "label": label, "reds": np.asarray(reds), "exits": exits, "entries": entries, "drafts": drafts,
        "forces": forces, "shares": shares, "status": status, "in_range": in_range, "temps": temps,
        "contact": np.sqrt(radii * drafts),
        "util": forces / caps_arr(), "max_util": float(np.max(forces / caps_arr())),
        "total": float(forces.sum()), "max": float(forces.max()), "top": top,
        "mean": float(forces.mean()), "cv": float(forces.std() / forces.mean()),
    }


def warnings_for(r: dict) -> list[tuple[str, str]]:
    """경고 메시지 목록 [(수준, 메시지)]"""
    msgs = []
    for i, s in enumerate(STANDS):
        f, u, cap = r["forces"][i], r["util"][i], LIMITS[s][1]
        if r["status"][i] == "danger":
            msgs.append(("danger", f"{s} 예측 하중 {f:.1f} MN (이용률 {u:.2f}) — 과거 최대 하중 {cap:.1f} MN 이상입니다"))
        elif r["status"][i] == "warn":
            msgs.append(("warn", f"{s} 예측 하중 {f:.1f} MN (이용률 {u:.2f}) — 이용률 {CAUTION_UTIL:.1f} 이상으로 과거 최대 하중에 가깝습니다"))
    t = r["top"]
    if r["shares"][t] >= CONCENTRATION_ALERT_SHARE:
        msgs.append(("warn", f"하중 집중: {STANDS[t]} 스탠드가 전체 하중의 {r['shares'][t] * 100:.1f}%를 차지합니다 "
                             f"(균등 배분 20%, 데모 기준 {CONCENTRATION_ALERT_SHARE * 100:.0f}%)."))
    for i, ok in enumerate(r["in_range"]):
        if not ok:
            msgs.append(("info", f"{STANDS[i]} 압하율 {r['reds'][i] * 100:.1f}%는 과거 운영 범위 밖입니다 — 예측 신뢰도 낮음."))
    return msgs


# =====================================================================================
# 6. AI Agent (규칙 기반 공정 분석)
# =====================================================================================
def analyze_with_llm(context: dict) -> str | None:
    """[LLM 연결 자리] 외부 LLM API를 붙일 때 이 함수만 구현한다.
    context에는 아래 analyze_process()가 쓰는 계산 결과가 모두 들어 있다.
    None을 돌려주면 규칙 기반 분석을 그대로 사용한다.
    """
    return None


def analyze_process(res: dict, inp: dict, cur: dict, opt: dict | None, opt_meta: dict | None) -> list[tuple[str, str]]:
    """계산된 수치만 근거로 5가지 질문에 답하는 규칙 기반 분석. [(질문, 답변 마크다운)]"""
    out = []
    t = cur["top"]
    ts = STANDS[t]
    order = np.argsort(-cur["forces"])
    second = int(order[1])

    # 1. 가장 하중이 집중된 스탠드
    diff = (cur["forces"][t] / cur["mean"] - 1) * 100
    out.append(("① 하중이 가장 집중된 스탠드", (
        f"**{ts}** 스탠드입니다. 예측 하중은 **{cur['forces'][t]:.1f} MN**으로 F1~F5 평균({cur['mean']:.1f} MN)보다 "
        f"**{diff:.0f}% 높고**, 전체 하중의 **{cur['shares'][t] * 100:.1f}**%를 차지합니다. "
        f"다음으로 높은 스탠드는 {STANDS[second]}({cur['forces'][second]:.1f} MN)입니다.")))

    # 2. 이유
    reasons = []
    r_med = res["red_med"][t] * 100
    r_now = cur["reds"][t] * 100
    if cur["reds"][t] >= np.max(cur["reds"]) - 1e-9:
        reasons.append(f"압하율 **{r_now:.1f}**%가 5개 스탠드 중 가장 큽니다")
    if r_now - r_med > 2:
        reasons.append(f"압하율이 {ts}의 과거 중앙값({r_med:.1f}%)보다 {r_now - r_med:.1f}%p 높습니다")
    if cur["drafts"][t] >= np.max(cur["drafts"]) - 1e-9:
        reasons.append(f"압하량 **{cur['drafts'][t]:.1f} mm**가 가장 커서 롤과 소재의 접촉 면적(접촉길이 {cur['contact'][t]:.0f} mm)이 큽니다")
    t_diff = cur["temps"][t] - res["temp_med"][t]
    if t_diff < -10:
        reasons.append(f"소재 온도 {cur['temps'][t] - 273.15:.0f}°C가 과거 중앙값보다 {abs(t_diff):.0f}°C 낮아 변형저항이 큽니다")
    if inp["width"] > res["width_med"] * 1.05:
        reasons.append(f"판 폭 {inp['width']:.0f} mm가 과거 중앙값({res['width_med']:.0f} mm)보다 넓어 하중이 커집니다")
    if not reasons:
        reasons.append(f"압하율 {r_now:.1f}%와 압하량 {cur['drafts'][t]:.1f} mm 조합이 다른 스탠드보다 하중을 더 크게 만듭니다")
    out.append(("② 하중이 높은 이유", "- " + "\n- ".join(reasons)))

    # 3. 문제점
    probs = []
    n_d, n_w = cur["status"].count("danger"), cur["status"].count("warn")
    if n_d or n_w:
        probs.append(f"위험 {n_d}개, 주의 {n_w}개 스탠드가 있습니다 (이용률 기준)")
    top2 = cur["forces"][order[:2]].sum() / cur["total"] * 100
    probs.append(f"상위 2개 스탠드({STANDS[int(order[0])]}, {STANDS[int(order[1])]})가 전체 하중의 {top2:.0f}%를 담당합니다")
    if cur["cv"] > res["cv_med"] * 1.1:
        probs.append(f"스탠드 간 하중 편차(CV {cur['cv']:.2f})가 과거 코일 중앙값({res['cv_med']:.2f})보다 큽니다")
    else:
        probs.append(f"스탠드 간 편차(CV {cur['cv']:.2f})는 과거 코일 중앙값({res['cv_med']:.2f}) 수준입니다. "
                     "F1 쏠림은 이 설비의 일반적인 패턴입니다")
    bad = [STANDS[i] for i, ok in enumerate(cur["in_range"]) if not ok]
    if bad:
        probs.append(f"{', '.join(bad)}의 압하율이 과거 운영 범위를 벗어나 예측 신뢰도가 낮습니다")
    out.append(("③ 현재 하중 분포의 문제점", "- " + "\n- ".join(probs)))

    # 4. 압하 배분 조정 방향
    if opt is not None:
        moves = []
        for i, s in enumerate(STANDS):
            d = (opt["reds"][i] - cur["reds"][i]) * 100
            if abs(d) >= 0.3:
                moves.append(f"{s} 압하율 {cur['reds'][i] * 100:.1f}% → {opt['reds'][i] * 100:.1f}% ({d:+.1f}%p)")
        text = ("총 압하량(최종 두께 " + f"{inp['final']:.2f} mm)은 유지하고 스탠드 간 배분만 조정합니다.\n- " + "\n- ".join(moves)) \
            if moves else "현재 배분이 제약조건 안에서 이미 최선에 가까워 추천할 변경이 없습니다."
    else:
        head = [i for i in range(5) if i != t and cur["reds"][i] < res["red_hi"][i] - 0.02]
        room = ", ".join(f"{STANDS[i]}(+{(res['red_hi'][i] - cur['reds'][i]) * 100:.0f}%p)" for i in head[:3])
        text = (f"{ts}의 압하량 일부를 후속 스탠드로 옮기는 방향이 적절합니다. 과거 운영 범위 기준으로 여유가 있는 스탠드: "
                f"{room if room else '없음'}. **'압하 배분 최적화'** 버튼을 누르면 제약조건 안에서 구체적인 배분을 계산합니다.")
    out.append(("④ 압하 배분 조정 방향", text))

    # 5. 최적화 후 변화
    if opt is None:
        out.append(("⑤ 최적화 후 변화", "아직 최적화를 실행하지 않았습니다."))
    else:
        dm = (opt["max"] / cur["max"] - 1) * 100
        du = (opt["max_util"] / cur["max_util"] - 1) * 100
        dcv = (opt["cv"] / cur["cv"] - 1) * 100
        ucur, uopt = int(np.argmax(cur["util"])), int(np.argmax(opt["util"]))
        lines = [f"최대 이용률 **{cur['max_util']:.2f} → {opt['max_util']:.2f} ({du:+.1f}%)** "
                 f"(이용률이 가장 높은 스탠드: {STANDS[ucur]} → {STANDS[uopt]})",
                 f"최대 하중 **{cur['max']:.1f} → {opt['max']:.1f} MN ({dm:+.1f}%)**",
                 f"하중 편차(CV) **{cur['cv']:.2f} → {opt['cv']:.2f} ({dcv:+.1f}%)**"]
        changed = [f"{s} {STATUS[a][0]}→{STATUS[b][0]}" for s, a, b in zip(STANDS, cur["status"], opt["status"]) if a != b]
        lines.append("경고 상태 변화: " + (", ".join(changed) if changed else "변화 없음"))
        if opt_meta and not opt_meta["feasible"]:
            lines.append("⚠ 이용률 1.0 미만을 모두 만족하는 배분을 찾지 못했습니다. 입력 조건 자체를 조정해야 할 수 있습니다.")
        if du > 0:
            lines.append("⚠ 최대 이용률이 늘었습니다. 이 모델과 제약조건에서는 현재 배분이 더 낫습니다.")
        elif dm > 0:
            lines.append("ℹ 최대 이용률은 줄었지만 최대 하중(MN)은 늘었습니다. 하중을 이용률 여유가 큰 스탠드로 옮겼기 때문이며, "
                         "이용률과 MN을 함께 봐야 합니다.")
        out.append(("⑤ 최적화 후 변화", "- " + "\n- ".join(lines)))
    return out


def stream_text(sections: list[tuple[str, str]]):
    """분석 결과를 타자 치듯 한 글자씩 내보낸다 (st.write_stream용)."""
    for title, body in sections:
        for chunk in f"**{title}**\n\n{body}\n\n".split(" "):
            yield chunk + " "
            time.sleep(0.004)


# =====================================================================================
# 7. 시각화
# =====================================================================================
def plotly_dark(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=14, color=C_TEXT)) if title else None,
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD, font=dict(color=C_MUTED, family="Malgun Gothic, sans-serif"),
        margin=dict(l=40, r=20, t=50 if title else 20, b=40), legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(gridcolor=C_BORDER, zeroline=False), yaxis=dict(gridcolor=C_BORDER, zeroline=False),
    )
    return fig


def fig_load_compare(cur: dict, opt: dict | None) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=STANDS, y=cur["forces"], name="현재", marker_color=C_STEEL,
                text=[f"{v:.1f}" for v in cur["forces"]], textposition="outside")
    if opt is not None:
        fig.add_bar(x=STANDS, y=opt["forces"], name="최적화 후", marker_color=C_ACCENT,
                    text=[f"{v:.1f}" for v in opt["forces"]], textposition="outside")
    caps = caps_arr()
    fig.add_scatter(x=STANDS, y=caps, mode="lines+markers", name="이용률 1.0 (과거 최대 하중)",
                    line=dict(color=C_DANGER, dash="dash", width=1.5), marker=dict(symbol="line-ew-open", size=18))
    fig.add_scatter(x=STANDS, y=CAUTION_UTIL * caps, mode="lines", name=f"이용률 {CAUTION_UTIL:.1f}",
                    line=dict(color=C_WARN, dash="dot", width=1.2))
    plotly_dark(fig, 340, "스탠드별 압연 하중 (MN) · 점선 = 스탠드별 과거 최대 하중 기준")
    fig.update_layout(barmode="group", yaxis_range=[0, max(caps.max(), cur["max"], opt["max"] if opt else 0) * 1.2])
    return fig


def fig_draft(cur: dict, opt: dict | None) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=STANDS, y=cur["drafts"], name="현재", marker_color=C_STEEL,
                text=[f"{d:.1f}" for d in cur["drafts"]], textposition="outside",
                customdata=cur["reds"] * 100, hovertemplate="%{x} 현재<br>압하량 %{y:.2f} mm<br>압하율 %{customdata:.1f}%<extra></extra>")
    if opt is not None:
        fig.add_bar(x=STANDS, y=opt["drafts"], name="최적화 후", marker_color=C_ACCENT,
                    text=[f"{d:.1f}" for d in opt["drafts"]], textposition="outside",
                    customdata=opt["reds"] * 100, hovertemplate="%{x} 최적화 후<br>압하량 %{y:.2f} mm<br>압하율 %{customdata:.1f}%<extra></extra>")
    plotly_dark(fig, 340, "스탠드별 압하량 배분 (mm) · 마우스를 올리면 압하율 표시")
    fig.update_layout(barmode="group", yaxis_range=[0, max(cur["drafts"].max(), opt["drafts"].max() if opt else 0) * 1.3])
    return fig


def fig_share(cur: dict, opt: dict | None) -> go.Figure:
    fig = go.Figure()
    rows = [("현재", cur)] + ([("최적화 후", opt)] if opt is not None else [])
    palette = ["#ffd9a0", "#ffb35c", "#ff8f2e", "#d9692a", "#a14a24"]
    for i, s in enumerate(STANDS):
        fig.add_bar(y=[n for n, _ in rows], x=[r["shares"][i] * 100 for _, r in rows], name=s, orientation="h",
                    marker_color=palette[i], text=[f"{r['shares'][i] * 100:.0f}%" for _, r in rows], textposition="inside")
    plotly_dark(fig, 220, "스탠드별 하중 비율 (%)")
    fig.update_layout(barmode="stack", xaxis_range=[0, 100], margin=dict(l=70, r=20, t=50, b=30),
                      legend=dict(orientation="h", y=1.18, x=0, traceorder="normal"),
                      yaxis=dict(autorange="reversed", gridcolor=C_BORDER))
    return fig


# ---- 3D 디지털 트윈 (장면은 twin/scene.html, 데이터 준비는 여기서) -----------------------------
TWIN_DIR = Path(__file__).resolve().parent / "twin"


def bottleneck_info(r: dict) -> dict:
    """병목 후보 스탠드: 이용률이 가장 높은 스탠드."""
    ratio = np.asarray(r["util"])
    i = int(np.argmax(ratio))
    level = "danger" if ratio[i] >= DANGER_UTIL else ("warn" if ratio[i] >= CAUTION_UTIL else "info")
    return {"stand": STANDS[i], "level": level,
            "text": f"이용률 {ratio[i]:.2f} ({r['forces'][i]:.1f} MN, 1.0 = 과거 최대 하중), 전체 하중의 {r['shares'][i] * 100:.0f}%"}


def energy_estimate(cur: dict, opt: dict | None) -> dict | None:
    """롤링 일량 추정(하중 x 접촉길이의 합)의 변화율. 에너지 실측 데이터가 없으므로 가정 기반 추정이다."""
    if opt is None:
        return None
    w_cur = float(np.sum(cur["forces"] * cur["contact"]))
    w_opt = float(np.sum(opt["forces"] * opt["contact"]))
    return {"delta_pct": (w_opt / w_cur - 1.0) * 100.0}


def build_twin_html(res: dict, inp: dict, cur: dict, opt: dict | None, actual=None, sample_id=None) -> str:
    """3D 디지털 트윈 화면(HTML)을 만든다.

    - 장면(애니메이션)은 twin/scene.html 이 브라우저 안에서 실행한다. Streamlit이 다시 그리지 않으므로 끊기지 않는다.
    - 소재 이동, 온도, 속도는 시뮬레이션 값이다. 하중과 압하 배분은 AI 예측(GBDT) 값이다.
    - 달궈진 소재에는 열 발광을 쓰지만, 롤과 스탠드와 상태 표시에는 빛 번짐(네온) 효과를 쓰지 않는다.
    """
    scene = (TWIN_DIR / "scene.html").read_text(encoding="utf-8")
    three = (TWIN_DIR / "three.min.js").read_text(encoding="utf-8")
    data = {
        "label": cur["label"], "grade": inp["grade"], "width": inp["width"],
        "actual": None if actual is None else [float(v) for v in actual], "sample_id": sample_id,
        "entries": [float(v) for v in cur["entries"]], "exits": [float(v) for v in cur["exits"]],
        "forces": [float(v) for v in cur["forces"]], "status": list(cur["status"]),
        "reds": [float(v) * 100 for v in cur["reds"]], "drafts": [float(v) for v in cur["drafts"]],
        "temps_c": [float(v) - 273.15 for v in cur["temps"]],
        "limits": [list(LIMITS[s]) for s in STANDS], "caps": [float(c) for c in caps_arr()],
        "opt": None if opt is None else {
            "forces": [float(v) for v in opt["forces"]], "reds": [float(v) * 100 for v in opt["reds"]],
            "drafts": [float(v) for v in opt["drafts"]], "status": list(opt["status"])},
        "warnings": [[lv, m] for lv, m in warnings_for(cur)],
        "bottleneck": bottleneck_info(cur), "energy": energy_estimate(cur, opt),
    }
    repl = {"__DATA__": json.dumps(data, ensure_ascii=False), "__BG__": C_BG, "__CARD__": C_CARD, "__TEXT__": C_TEXT,
            "__MUTED__": C_MUTED, "__BORDER__": C_BORDER, "__ACCENT__": C_ACCENT,
            "__OK__": C_OK, "__WARN__": C_WARN, "__DANGER__": C_DANGER}
    for k, v in repl.items():
        scene = scene.replace(k, v)
    return scene.replace("__THREE__", three)     # three.js는 마지막에 넣는다 (본문 치환에 섞이지 않게)


def kpi_card(title: str, value: str, unit: str, sub: str, color: str, after: str | None = None) -> str:
    after_html = f"<div class='kpi-after'>최적화 후 <b>{after}</b></div>" if after else ""
    return (f"<div class='kpi' style='border-top:3px solid {color}'><div class='kpi-t'>{title}</div>"
            f"<div class='kpi-v'>{value}<span class='kpi-u'>{unit}</span></div><div class='kpi-s'>{sub}</div>{after_html}</div>")


def stand_card(i: int, cur: dict, opt: dict | None) -> str:
    s = STANDS[i]
    danger = LIMITS[s][1]
    name, color = STATUS[cur["status"][i]]
    pct = min(cur["forces"][i] / danger * 100, 100)
    html = (f"<div class='sc' style='border-left:4px solid {color}'><div class='sc-h'><b>{s}</b>"
            f"<span class='badge' style='background:{color}22;color:{color};border:1px solid {color}'>{name}</span></div>"
            f"<div class='sc-v'>{cur['forces'][i]:.1f}<span class='kpi-u'>MN</span></div>"
            f"<div class='sc-o' style='margin:-4px 0 4px 0'>이용률 <b style='color:{color}'>{cur['util'][i]:.2f}</b></div>"
            f"<div class='bar'><div style='width:{pct:.0f}%;background:{color}'></div></div>")
    if opt is not None:
        n2, c2 = STATUS[opt["status"][i]]
        d = (opt["forces"][i] / cur["forces"][i] - 1) * 100
        html += (f"<div class='sc-o'>→ {opt['forces'][i]:.1f} MN · 이용률 {opt['util'][i]:.2f}<br>({d:+.1f}%) "
                 f"<span style='color:{c2}'>● {n2}</span></div>")
    return html + "</div>"


def fig_mini(cur: dict, opt: dict | None) -> go.Figure:
    """왼쪽 패널용 작은 하중 곡선 (F1 -> F5)."""
    fig = go.Figure()
    fig.add_scatter(x=STANDS, y=cur["forces"], mode="lines+markers", name="현재", line=dict(color=C_STEEL, width=2))
    if opt is not None:
        fig.add_scatter(x=STANDS, y=opt["forces"], mode="lines+markers", name="최적화 후", line=dict(color=C_ACCENT, width=2))
    fig.add_scatter(x=STANDS, y=caps_arr(), mode="lines", name="이용률 1.0", line=dict(color=C_DANGER, dash="dash", width=1))
    plotly_dark(fig, 190)
    fig.update_layout(title=dict(text=""), margin=dict(l=30, r=8, t=24, b=24), legend=dict(orientation="h", y=1.25, x=0, font=dict(size=10)),
                      yaxis_range=[0, max(caps_arr().max(), cur["max"], opt["max"] if opt else 0) * 1.15], paper_bgcolor="#121216", plot_bgcolor="#121216")
    return fig


def render_actual_compare(cur: dict, actual: np.ndarray, smp: dict):
    """샘플 코일의 실측 하중과 모델 예측을 나란히 보여 준다."""
    err = cur["forces"] - actual
    caps = caps_arr()
    st.markdown(f"<div class='panel-t'>샘플 코일 실측 비교 <span>ACTUAL VS PREDICTED</span> · 코일 {smp['id']} · 연주기 {smp['caster']}</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    fig = go.Figure()
    fig.add_bar(x=STANDS, y=actual, name="실측", marker_color="#cfcfd6", text=[f"{v:.1f}" for v in actual], textposition="outside")
    fig.add_bar(x=STANDS, y=cur["forces"], name="예측 (GBDT)", marker_color=C_STEEL,
                text=[f"{v:.1f}" for v in cur["forces"]], textposition="outside")
    plotly_dark(fig, 300, "스탠드별 하중 (MN)")
    fig.update_layout(barmode="group", yaxis_range=[0, max(actual.max(), cur["max"]) * 1.25])
    c1.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(pd.DataFrame({"스탠드": STANDS, "실측 (MN)": np.round(actual, 1), "예측 (MN)": np.round(cur["forces"], 1),
                                   "오차 (MN)": np.round(err, 1), "실측 이용률": np.round(actual / caps, 2),
                                   "예측 이용률": np.round(cur["util"], 2)}), hide_index=True, width="stretch")
        st.markdown(f"<div class='note'>평균 절대 오차(MAE) <b>{np.abs(err).mean():.2f} MN</b> · 최대 오차 {np.abs(err).max():.2f} MN<br>"
                    f"최대 이용률: 실측 <b>{(actual / caps).max():.2f}</b> / 예측 <b>{cur['max_util']:.2f}</b><br>"
                    "예측 모델: GBDT (단조 제약). 처음 보는 코일 기준 평균 오차 0.44 MN.</div>", unsafe_allow_html=True)


def sidebar_summary(cur: dict, opt: dict | None):
    """왼쪽 패널 아래쪽: 핵심 숫자 3개와 작은 곡선 그래프."""
    def big(title, v, unit, v2, fmt):
        arrow = ""
        if v2 is not None:
            arrow = f"<i class='{'dn' if v2 < v else 'up'}'>{'↓' if v2 < v else '↑'} {fmt.format(v2)}</i>"
        return (f"<div class='sb-kpi'><div class='sb-kt'>{title}</div>"
                f"<div class='sb-kv'>{fmt.format(v)}<span>{unit}</span>{arrow}</div></div>")
    with st.sidebar:
        st.markdown("<hr class='sb-hr'>", unsafe_allow_html=True)
        st.markdown(big("최대 압연 하중", cur["max"], "MN", opt["max"] if opt else None, "{:.1f}")
                    + big("최대 이용률", cur["max_util"], "", opt["max_util"] if opt else None, "{:.2f}")
                    + big("하중 편차 (CV)", cur["cv"], "", opt["cv"] if opt else None, "{:.2f}"), unsafe_allow_html=True)
        st.markdown("<div class='sb-kt' style='margin-top:10px'>압연 하중 곡선 (F1→F5)</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_mini(cur, opt), width="stretch", config={"displayModeBar": False})


CSS = f"""
<style>
.stApp {{ background: {C_BG}; color: {C_TEXT}; }}
[data-testid="stSidebar"] {{ background: #121216; border-right: 1px solid {C_BORDER}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
.topbar {{ display:flex; align-items:center; justify-content:space-between; padding: 6px 2px 10px 2px; }}
.tb-title {{ color:{C_MUTED}; font-size:0.8rem; letter-spacing:0.14em; }}
.sb-team {{ display:inline-block; margin:0 0 10px 0; padding:3px 14px; border:1px solid {C_ACCENT}; border-radius:999px;
  color:{C_ACCENT}; font-size:0.9rem; font-weight:700; letter-spacing:0.18em; }}
.sb-title {{ color:{C_MUTED}; font-size:0.66rem; letter-spacing:0.12em; line-height:1.5; margin-bottom:2px; }}
.sb-brand {{ font-size:1.35rem; font-weight:700; letter-spacing:0.12em; margin-bottom:10px; }}
.sb-brand span {{ color:{C_ACCENT}; }}
.sb-hr {{ border:none; border-top:1px solid {C_BORDER}; margin:14px 0 10px 0; }}
.sb-kpi {{ padding:6px 0 10px 0; border-bottom:1px solid {C_BORDER}; }}
.sb-kt {{ color:{C_MUTED}; font-size:0.78rem; }}
.sb-kv {{ font-size:2.0rem; font-weight:700; color:{C_TEXT}; line-height:1.3; }}
.sb-kv span {{ font-size:0.9rem; color:{C_MUTED}; margin-left:4px; font-weight:400; }}
.sb-kv i {{ font-size:1.0rem; font-style:normal; margin-left:10px; }}
.sb-kv i.dn {{ color:{C_OK}; }}
.sb-kv i.up {{ color:{C_DANGER}; }}
.tag {{ display:inline-block; padding:3px 10px; margin-left:6px; border-radius:999px; font-size:0.72rem;
  border:1px solid {C_WARN}; color:{C_WARN}; background:#f5c54214; }}
.tag.live {{ border-color:{C_ACCENT}; color:{C_ACCENT}; background:#ff8f2e14; }}
.kpi {{ background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:12px; padding:14px 16px; min-height:132px; }}
.kpi-t {{ color:{C_MUTED}; font-size:0.78rem; letter-spacing:0.05em; }}
.kpi-v {{ font-size:1.9rem; font-weight:700; color:{C_TEXT}; line-height:1.25; }}
.kpi-u {{ font-size:0.85rem; color:{C_MUTED}; margin-left:4px; font-weight:400; }}
.kpi-s {{ color:{C_MUTED}; font-size:0.8rem; }}
.kpi-after {{ margin-top:6px; padding-top:6px; border-top:1px dashed {C_BORDER}; color:{C_ACCENT}; font-size:0.82rem; }}
.sc {{ background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:10px; padding:10px 12px; }}
.sc-h {{ display:flex; justify-content:space-between; align-items:center; color:{C_TEXT}; }}
.sc-v {{ font-size:1.45rem; font-weight:700; margin:2px 0 6px 0; }}
.sc-o {{ margin-top:6px; font-size:0.8rem; color:{C_MUTED}; }}
.badge {{ font-size:0.7rem; padding:1px 8px; border-radius:999px; }}
.bar {{ height:6px; background:#26262d; border-radius:4px; overflow:hidden; }}
.bar div {{ height:100%; border-radius:4px; }}
.alert {{ padding:8px 12px; border-radius:8px; margin:6px 0; font-size:0.86rem; }}
.alert.danger {{ background:#ef444418; border:1px solid {C_DANGER}; color:#fecaca; }}
.alert.warn {{ background:#f5c54216; border:1px solid {C_WARN}; color:#fde68a; }}
.alert.info {{ background:#4f8ccf18; border:1px solid {C_STEEL}; color:#bfdbfe; }}
.alert.ok {{ background:#22c55e16; border:1px solid {C_OK}; color:#bbf7d0; }}
.panel-t {{ font-size:1.0rem; font-weight:600; color:{C_TEXT}; margin:10px 0 6px 0; letter-spacing:0.04em; }}
.panel-t span {{ color:{C_ACCENT}; }}
.note {{ color:{C_MUTED}; font-size:0.78rem; }}
</style>
"""


# =====================================================================================
# 8. 화면 (Streamlit)
# =====================================================================================
def init_state(res: dict):
    """입력 위젯의 초기값과 시뮬레이션 상태를 session_state에 준비한다."""
    rep = res["rep"]
    defaults = {"in_grade": rep["grade"], "in_width": int(round(rep["width"])), "in_entry": round(rep["entry"], 1),
                "in_final": round(rep["final"], 2), "in_temp": int(round(rep["temp_c"])), "in_manual": False,
                "in_sample": 0, "sample": None, "cur": None, "opt": None, "opt_meta": None,
                "anim": None, "agent_new": False}
    reds = default_reductions(res, rep["entry"], rep["final"])
    for i in range(4):
        defaults[f"in_r{i + 1}"] = float(round(np.clip(reds[i], res["red_lo"][i], res["red_hi"][i]) * 100, 1))
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def apply_preset(name: str, res: dict):
    """시나리오 버튼: 입력값을 미리 정해 둔 조건으로 바꾼다 (버튼 콜백)."""
    rep = res["rep"]
    ss = st.session_state
    ss.update(in_grade=rep["grade"], in_width=int(round(rep["width"])), in_entry=round(rep["entry"], 1),
              in_final=round(rep["final"], 2), in_temp=int(round(rep["temp_c"])))
    if name == "표준":
        ss["in_manual"] = False
        reds = default_reductions(res, ss["in_entry"], ss["in_final"])
        for i in range(4):
            ss[f"in_r{i + 1}"] = float(round(np.clip(reds[i], res["red_lo"][i], res["red_hi"][i]) * 100, 1))
    else:  # "F1 집중": F1 압하율을 관측 범위 상단 근처로 올리고 나머지에 총 변형률을 나눠 준다
        r1 = float(res["red_hi"][0]) * 0.97
        rest = math.log(ss["in_entry"] / ss["in_final"]) + math.log(1 - r1)
        w = res["strain_share"][1:] / res["strain_share"][1:].sum()
        eps = w * rest
        reds = np.concatenate([[r1], 1 - np.exp(-eps)])
        ss["in_manual"] = True
        ss["in_width"] = int(min(res["width_range"][1], round(rep["width"]) + 170))   # 폭이 넓은 코일: F1 이용률이 0.9를 넘도록
        for i in range(4):
            ss[f"in_r{i + 1}"] = float(round(np.clip(reds[i], res["red_lo"][i], res["red_hi"][i]) * 100, 1))
    ss["cur"] = ss["opt"] = ss["opt_meta"] = None   # 이전 결과는 지우고 새로 계산
    ss["anim"] = None
    ss["in_sample"], ss["sample"] = 0, None         # 프리셋을 쓰면 샘플 코일 선택은 해제한다


def load_sample(res: dict):
    """샘플 코일 선택 콜백: 실제 코일의 조건과 압하 배분(F1~F4)을 입력칸에 채우고, 실측 하중을 기억해 둔다."""
    ss = st.session_state
    cid = ss["in_sample"]
    ss["cur"] = ss["opt"] = ss["opt_meta"] = None
    ss["anim"] = None
    if not cid:
        ss["sample"] = None
        return
    c = res["coils"][cid]
    ss.update(in_grade=c["grade"], in_width=int(round(c["width"])), in_entry=round(c["entry"], 2),
              in_final=round(c["final"], 2), in_temp=int(round(c["temp_c"])), in_manual=True)
    reds4 = [round(float(c["reds"][i]) * 100, 4) for i in range(4)]
    for i in range(4):
        ss[f"in_r{i + 1}"] = reds4[i]
    ss["sample"] = {"id": cid, "caster": c["caster"], "actual": [float(v) for v in c["forces"]],
                    "radii": [float(v) for v in c["radii"]], "temps": [float(v) for v in c["temps"]],
                    "exact": dict(width=c["width"], entry=c["entry"], final=c["final"]),   # 슬라이더 반올림 전 원래 값
                    "inputs": dict(grade=c["grade"], width=float(ss["in_width"]), entry=ss["in_entry"], final=ss["in_final"],
                                   temp_c=float(ss["in_temp"]), reds4=[v / 100 for v in reds4])}


def sample_matches(inp: dict, smp: dict | None) -> bool:
    """현재 입력이 불러온 샘플 코일 조건과 같은지 (같을 때만 실측과 비교할 수 있다)."""
    if not smp or not inp["manual"]:
        return False
    i = smp["inputs"]
    return bool(inp["grade"] == i["grade"] and abs(inp["width"] - i["width"]) < 0.5 and abs(inp["entry"] - i["entry"]) < 0.005
                and abs(inp["final"] - i["final"]) < 0.005 and abs(inp["temp_c"] - i["temp_c"]) < 0.5
                and np.allclose(inp["reds_in"], i["reds4"], atol=1e-4))


def sidebar_inputs(res: dict) -> dict:
    ss = st.session_state
    with st.sidebar:
        st.markdown("<div class='sb-team'>great science</div>"
                    "<div class='sb-title'>INTELLIGENT HOT ROLLING<br>DIGITAL TWIN SYSTEM</div>"
                    "<div class='sb-brand'>STEEL <span>FLOW</span> AI</div>"
                    "<div class='panel-t'>공정 조건 <span>INPUT</span></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("표준 조건", on_click=apply_preset, args=("표준", res), width="stretch")
        c2.button("F1 집중", on_click=apply_preset, args=("F1", res), width="stretch")
        st.selectbox("샘플 코일 불러오기", [0] + list(SAMPLE_COILS), key="in_sample", on_change=load_sample, args=(res,),
                     format_func=lambda c: "(직접 입력)" if not c else f"코일 {c} · {SAMPLE_COILS[c]}",
                     help="실제 조업 코일의 조건과 압하 배분을 불러옵니다. 불러온 조건을 바꾸지 않으면 실측 하중과 예측을 비교해 보여 줍니다.")
        if ss.get("sample"):
            st.caption(f"연주기 {ss['sample']['caster']} · 롤 반경과 스탠드별 온도는 이 코일의 실측값을 씁니다. "
                       "예측 모델은 연주기를 입력으로 쓰지 않습니다.")
            st.button("↺ 이 코일 조건으로 되돌리기", on_click=load_sample, args=(res,), width="stretch",
                      help="슬라이더를 바꾼 뒤 불러온 코일의 원래 조건으로 되돌립니다.")

        grade = st.selectbox("강종", res["grades"], key="in_grade",
                             format_func=lambda g: f"{g}  (코일 {res['grade_counts'][g]}개)")
        if res["grade_counts"][grade] < 200:
            st.caption("이 강종은 학습 데이터가 적어 예측이 덜 정확할 수 있습니다.")
        width = st.slider("소재 폭 (mm)", int(res["width_range"][0]), int(res["width_range"][1]), key="in_width")
        entry = st.slider("입측 두께 (mm)", round(res["entry_range"][0], 1), round(res["entry_range"][1], 1),
                          step=0.1, key="in_entry")
        final = st.slider("목표 출측 두께 (mm)", round(res["final_range"][0], 2), round(res["final_range"][1], 2),
                          step=0.05, key="in_final")
        temp = st.slider("F1 압연 온도 (°C)", int(res["temp1_range_c"][0]), int(res["temp1_range_c"][1]), key="in_temp")

        manual = st.toggle("현재 압하 배분 직접 입력", key="in_manual",
                           help="끄면 과거 코일의 평균적인 배분 규칙으로 현행 배분을 자동 설정합니다.")
        reds_in = None
        if manual:
            st.caption("F1~F4를 정하면 F5는 목표 두께에 맞춰 자동 계산됩니다.")
            reds_in = [st.slider(f"{STANDS[i]} 압하율 (%)", round(res["red_min"][i] * 100, 1),
                                 round(res["red_max"][i] * 100, 1), step=0.5, key=f"in_r{i + 1}") / 100 for i in range(4)]
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶  시뮬레이션 실행", type="primary", width="stretch")
        optimize = st.button("⚙  압하 배분 최적화", width="stretch")
    return dict(grade=grade, width=float(width), entry=float(entry), final=float(final), temp_c=float(temp),
                manual=manual, reds_in=reds_in, run=run, optimize=optimize)


def validate(inp: dict, res: dict) -> tuple[bool, str, np.ndarray | None]:
    """입력 검증과 현재 압하율(5개) 계산. (정상 여부, 오류 메시지, 압하율)"""
    if inp["final"] >= inp["entry"]:
        return False, "목표 출측 두께는 입측 두께보다 작아야 합니다.", None
    if inp["manual"]:
        r4 = np.array(inp["reds_in"])
        r5 = 1.0 - inp["final"] / (inp["entry"] * float(np.prod(1.0 - r4)))
        if r5 <= 0:
            return False, (f"F1~F4에서 이미 목표 두께({inp['final']:.2f} mm)보다 얇아져 F5 압하율이 {r5 * 100:.1f}%가 됩니다. "
                           "F1~F4 압하율을 낮추거나 목표 두께를 키워 주세요."), None
        if r5 >= 0.7:
            return False, f"F5 압하율이 {r5 * 100:.1f}%로 비현실적입니다. F1~F4 압하율을 높여 주세요.", None
        return True, "", np.append(r4, r5)
    return True, "", default_reductions(res, inp["entry"], inp["final"])


def main():
    st.set_page_config(page_title="STEEL FLOW AI", page_icon="⚙", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    res = load_resources()
    if "error" in res:
        st.error(res["error"])
        st.stop()
    apply_caps(res)                  # 경고 기준(이용률)을 데이터에서 계산한 값으로 갱신
    init_state(res)
    ss = st.session_state

    # ---- 헤더
    model_tag = f"<span class='tag{'' if MODEL_IS_PLACEHOLDER else ' live'}'>예측 모델: {MODEL_NAME}</span>"
    limit_tag = "<span class='tag'>경고 기준: 과거 최대 하중 대비 이용률</span>"
    st.markdown(f"<div class='topbar'><span class='tb-title'>HOT FINISHING MILL · F1~F5 DIGITAL TWIN</span>"
                f"<div>{model_tag}{limit_tag}<span class='tag live'>최적화: 과거 운영 범위 안 탐색</span></div></div>",
                unsafe_allow_html=True)

    inp = sidebar_inputs(res)
    smp = ss.get("sample")
    matched = sample_matches(inp, smp)   # 불러온 코일 조건 그대로인지 (그럴 때만 실측과 비교)
    if smp:   # 불러온 코일의 폭·두께를 그대로 두었으면 슬라이더 반올림 전 원래 값으로 계산한다 (발표 숫자와 일치)
        i = smp["inputs"]
        if abs(inp["width"] - i["width"]) < 0.5 and abs(inp["entry"] - i["entry"]) < 0.005 and abs(inp["final"] - i["final"]) < 0.005:
            inp.update(smp["exact"])
    ok, err, cur_reds = validate(inp, res)
    if not ok:
        st.error(err)
        st.stop()

    # ---- 실행 버튼 처리
    inp["radii"], inp["temps"] = stand_conditions(res, inp, smp)
    inp["caster"] = smp["caster"] if smp else "S"
    sig = (inp["grade"], inp["width"], inp["entry"], inp["final"], inp["temp_c"], tuple(np.round(cur_reds, 4)))
    if inp["run"] or inp["optimize"] or ss["cur"] is None:
        if ss["cur"] is None or inp["run"] or ss["cur"]["sig"] != sig:
            ss["cur"] = build_result(res, inp, cur_reds, "현재")
            ss["cur"]["sig"] = sig
            ss["opt"], ss["opt_meta"] = None, None
            ss["anim"] = "cur" if (inp["run"] or inp["optimize"]) else None
            ss["agent_new"] = bool(inp["run"] or inp["optimize"])
    if inp["optimize"]:
        with st.spinner("제약조건 안에서 압하 배분을 탐색하는 중..."):
            meta = optimize_schedule(res, inp, cur_reds)
        if "error" in meta:
            st.warning(f"최적화할 수 없습니다: {meta['error']}")
        else:
            ss["opt"] = build_result(res, inp, meta["reductions"], "최적화 후")
            ss["opt_meta"] = meta
            ss["anim"], ss["agent_new"] = "opt", True

    cur, opt, opt_meta = ss["cur"], ss["opt"], ss["opt_meta"]
    actual = np.array(smp["actual"]) if (smp and cur["sig"] == sig and matched) else None   # 불러온 조건 그대로일 때만 실측 비교
    if cur["sig"] != sig:
        st.markdown("<div class='alert info'>입력값이 바뀌었습니다. 아래 결과는 이전 조건 기준입니다. "
                    "<b>시뮬레이션 실행</b>을 눌러 갱신하세요.</div>", unsafe_allow_html=True)

    sidebar_summary(cur, opt)

    # ---- KPI
    def after(v, fmt="{:.1f}", pct=True):
        if opt is None:
            return None
        return fmt.format(v) + (f" ({(v / base - 1) * 100:+.1f}%)" if pct else "")

    t = cur["top"]
    k = st.columns(5)
    worst = "danger" if "danger" in cur["status"] else ("warn" if "warn" in cur["status"] else "ok")
    base = cur["total"]
    k[0].markdown(kpi_card("전체 압연 하중 (ΣF1~F5)", f"{cur['total']:.1f}", "MN", f"스탠드 {len(STANDS)}개 합계", C_STEEL,
                           after(opt["total"]) if opt else None), unsafe_allow_html=True)
    base = cur["max"]
    k[1].markdown(kpi_card("최대 압연 하중", f"{cur['max']:.1f}", "MN", f"{STANDS[t]} · 이용률 {cur['util'][t]:.2f} · {STATUS[cur['status'][t]][0]}",
                           STATUS[cur["status"][t]][1], after(opt["max"]) if opt else None), unsafe_allow_html=True)
    k[2].markdown(kpi_card("하중 집중 스탠드", STANDS[t], "", f"전체의 {cur['shares'][t] * 100:.1f}% (균등 20%)", C_ACCENT,
                           f"{STANDS[opt['top']]} · {opt['shares'][opt['top']] * 100:.1f}%" if opt else None), unsafe_allow_html=True)
    base = cur["mean"]
    k[3].markdown(kpi_card("F1~F5 평균 하중", f"{cur['mean']:.1f}", "MN", "스탠드 평균", C_STEEL,
                           after(opt["mean"]) if opt else None), unsafe_allow_html=True)
    base = cur["cv"]
    k[4].markdown(kpi_card("하중 편차 (CV)", f"{cur['cv']:.2f}", "", f"과거 코일 중앙값 {res['cv_med']:.2f}", C_OK if cur["cv"] <= res["cv_med"] * 1.1 else C_WARN,
                           after(opt["cv"], "{:.2f}") if opt else None), unsafe_allow_html=True)

    # ---- 3D 디지털 트윈 (애니메이션은 브라우저 안에서 계속 실행되며, 이 화면은 다시 그려지지 않는다)
    st.markdown(f"<div class='panel-t'>제철소 디지털 트윈 <span>3D LIVE SIMULATION</span> · {cur['label']} 조건"
                f"{' · 최적화 후 비교 포함' if opt is not None else ''}</div>", unsafe_allow_html=True)
    twin_html = build_twin_html(res, inp, cur, opt, actual, smp["id"] if actual is not None else None)   # 우리가 만든 HTML만 삽입한다
    if hasattr(st, "iframe"):          # 최신 Streamlit
        st.iframe(twin_html, height="content")
    else:                              # 구버전 호환
        components.html(twin_html, height=1250)
    ss["anim"] = None
    st.caption("3D 장면의 소재 이동, 온도, 속도는 시뮬레이션 값이며 실측 데이터가 아닙니다. "
               "F1~F5 하중과 압하 배분은 AI 예측(GBDT) 값이고, 이용률은 과거 최대 하중 대비 값(실제 설비 정격 아님)입니다.")
    if actual is not None:
        render_actual_compare(cur, actual, smp)
    elif smp:
        st.markdown(f"<div class='alert info'>코일 {smp['id']}의 조건이 바뀌어 실측과 비교할 수 없습니다. 바꾼 조건의 하중은 모델 예측값입니다.</div>",
                    unsafe_allow_html=True)

    # ---- 설비 경고 (위) / AI 공정 분석 (아래): 서로 다른 구역으로 나눠서 위아래로 둔다
    with st.container():
        st.markdown("<div class='panel-t'>설비 경고 <span>WARNING</span></div>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i in range(5):
            cols[i].markdown(stand_card(i, cur, opt), unsafe_allow_html=True)
        msgs = warnings_for(cur)
        if not msgs:
            st.markdown("<div class='alert ok'>모든 스탠드가 정상 범위입니다.</div>", unsafe_allow_html=True)
        for level, m in msgs:
            st.markdown(f"<div class='alert {level}'>{'🔴' if level == 'danger' else '🟡' if level == 'warn' else 'ℹ'} {m}</div>",
                        unsafe_allow_html=True)
        if opt is not None:
            before = {"danger": cur["status"].count("danger"), "warn": cur["status"].count("warn")}
            afterc = {"danger": opt["status"].count("danger"), "warn": opt["status"].count("warn")}
            st.markdown(f"<div class='alert info'>최적화 후 경고: 위험 {before['danger']}→{afterc['danger']}개, "
                        f"주의 {before['warn']}→{afterc['warn']}개 · 탐색 후보 {opt_meta['n_evaluated']:,}개 평가</div>",
                        unsafe_allow_html=True)
        caps_txt = " / ".join(f"{s} {c:.1f}" for s, c in zip(STANDS, caps_arr()))
        st.markdown(f"<div class='note'>※ 이용률 = 예측 하중 ÷ 스탠드별 과거 최대 하중(데이터 상위 0.5%). 주의 {CAUTION_UTIL:.1f} 이상, 위험 {DANGER_UTIL:.1f} 이상. "
                    f"과거 최대 하중(MN): {caps_txt}. 실제 설비 정격이 아니므로 하중(MN)과 이용률을 함께 보세요.</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='panel-t'>AI 공정 분석 <span>AGENT</span></div>", unsafe_allow_html=True)
        ctx = dict(res=res, inp=inp, cur=cur, opt=opt, opt_meta=opt_meta)
        llm_text = analyze_with_llm(ctx)
        sections = analyze_process(res, inp, cur, opt, opt_meta) if llm_text is None else [("AI 분석", llm_text)]
        box = st.container(border=True)
        with box:
            if ss["agent_new"]:
                st.write_stream(stream_text(sections))   # 새 결과일 때만 타자 효과
                ss["agent_new"] = False
            else:
                for title, body in sections:
                    st.markdown(f"**{title}**\n\n{body}\n")
        st.markdown("<div class='note'>※ 규칙 기반 분석입니다. 위 수치는 모두 계산된 값이며, 예측 모델이 교체되면 함께 갱신됩니다.</div>",
                    unsafe_allow_html=True)

    # ---- 그래프
    st.markdown("<div class='panel-t'>하중 · 압하 배분 분석 <span>ANALYSIS</span></div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    g1.plotly_chart(fig_load_compare(cur, opt), width="stretch", config={"displayModeBar": False})
    g2.plotly_chart(fig_draft(cur, opt), width="stretch", config={"displayModeBar": False})
    st.plotly_chart(fig_share(cur, opt), width="stretch", config={"displayModeBar": False})

    # ---- 상세 표
    with st.expander("스탠드별 상세 데이터"):
        rows = []
        for r in ([cur] + ([opt] if opt else [])):
            for i, s in enumerate(STANDS):
                rows.append({"구분": r["label"], "스탠드": s, "입구 두께 (mm)": round(r["entries"][i], 2),
                             "출구 두께 (mm)": round(r["exits"][i], 2), "압하량 (mm)": round(r["drafts"][i], 2),
                             "압하율 (%)": round(r["reds"][i] * 100, 1), "예측 하중 (MN)": round(r["forces"][i], 2),
                             "하중 비율 (%)": round(r["shares"][i] * 100, 1), "상태": STATUS[r["status"][i]][0],
                             "학습 범위": "안" if r["in_range"][i] else "밖"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if opt_meta is not None:
        st.markdown(f"<div class='note'>최적화(khj/optimize.py): 최종 두께 유지, 모든 스탠드 압하율은 과거 운영 범위(1~99%) 안. "
                    f"범위 안 무작위 후보 {opt_meta['n_evaluated']:,}개를 GBDT로 예측해 '최대 이용률 + 0.5 × 이용률 편차'가 가장 작은 배분을 고릅니다. "
                    "수학적 최적해를 보장하지는 않습니다.</div>",
                    unsafe_allow_html=True)


main()
