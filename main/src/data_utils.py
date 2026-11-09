"""열연 압연 하중 데이터 불러오기와 정리 (팀 공통 정리 기준).

원본: pjh/data_utils.py (박지해). main으로 옮기면서 CSV 경로만 바꿨다.

사용:
    from data_utils import load_clean
    df, info = load_clean()
"""
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "raw_data" / "metals.csv"   # main/raw_data/metals.csv
STANDS = ["F1", "F2", "F3", "F4", "F5"]
TOL_GAUGE = 0.01  # mm, 앞 스탠드 출구와 다음 스탠드 입구 두께 허용 오차
TOL_WIDTH = 0.01  # mm, 한 코일 안 폭 허용 변동
FORCE = "Force (N)"


def load_raw(path=CSV):
    """CSV 뒤의 빈 컬럼 18개를 버리고 앞 10개 컬럼만 읽는다."""
    return pd.read_csv(path, usecols=range(10))


def find_broken_coils(df, tol_gauge=TOL_GAUGE, tol_width=TOL_WIDTH):
    """연속성이 깨진 코일을 사유별로 {사유: 코일 ID 집합}으로 반환한다."""
    d = df.sort_values(["Coil ID", "Stand ID"])
    g = d.groupby("Coil ID")
    stand_ok = g["Stand ID"].apply(lambda s: list(s) == STANDS)
    gap = (d["Exit gauge (mm)"] - g["Entry gauge (mm)"].shift(-1)).abs()
    wr = g["Width (mm)"].max() - g["Width (mm)"].min()
    return {
        "스탠드 구성 불완전": set(stand_ok.index[~stand_ok]),
        "두께 불연속": set(d.loc[gap > tol_gauge, "Coil ID"]),
        "폭 변동": set(wr.index[wr > tol_width]),
    }


def add_features(df):
    """파생 변수를 추가한 복사본을 반환한다."""
    d = df.copy()
    d["draft"] = d["Entry gauge (mm)"] - d["Exit gauge (mm)"]
    d["reduction"] = d["draft"] / d["Entry gauge (mm)"]
    d["strain"] = np.log(d["Entry gauge (mm)"] / d["Exit gauge (mm)"])
    d["contact_len"] = np.sqrt(d["Radius (mm)"] * d["draft"])
    d["force_MN"] = d[FORCE] / 1e6
    return d


def load_clean():
    """(정리된 DataFrame, 제외 사유별 코일 수 dict)를 반환한다."""
    df = load_raw()
    reasons = find_broken_coils(df)
    broken = set().union(*reasons.values())
    clean = df[~df["Coil ID"].isin(broken)].reset_index(drop=True)
    info = {k: len(v) for k, v in reasons.items()}
    info["제외 코일(중복 제거)"] = len(broken)
    return clean, info
