"""원본 엑셀(Nucor 사상압연 데이터)을 분석용 CSV로 정리한다.

원본 xlsx는 'Strict OOXML' 형식이라 openpyxl로 열리지 않아 calamine 엔진을 쓴다.
실행: python src/prepare_data.py
출력: data/processed/rolling.csv (스탠드 단위 long 형식)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/metals-3957876-supplementary.xlsx"
OUT = ROOT / "data/processed/rolling.csv"

COLS = {
    "Grade": "grade",
    "Coil ID": "coil",
    "Caster ID": "caster",
    "Stand ID": "stand",
    "Entry gauge (mm)": "h_in",
    "Exit gauge (mm)": "h_out",
    "Radius (mm)": "R",
    "Width (mm)": "W",
    "Temperature (K)": "T_K",
    "Force (N)": "F_N",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW, sheet_name="All Grades Data", engine="calamine")
    return df.rename(columns=COLS)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stand_no"] = df["stand"].str[1].astype(int)
    df["dh"] = df["h_in"] - df["h_out"]                  # 압하량 (mm)
    df["reduction"] = df["dh"] / df["h_in"]              # 압하율
    df["L"] = np.sqrt(df["R"] * df["dh"])                # 접촉 길이 (mm)
    df["T_C"] = df["T_K"] - 273.15
    df["F_MN"] = df["F_N"] / 1e6
    df["k_mean"] = df["F_N"] / (df["W"] * df["L"])       # 평균 변형 저항 근사 (MPa)
    return df


def add_chain_flag(df: pd.DataFrame) -> pd.DataFrame:
    """코일 단위 품질 플래그: 스탠드 5개가 모두 있고 F(k) 입측 = F(k-1) 출측이면 True.
    하중 예측(행 단위)에는 전체를, 압하 배분 분석·최적화(코일 단위)에는 True만 쓴다."""
    s = df.sort_values(["coil", "stand_no"])
    gap = (s["h_in"] - s.groupby("coil")["h_out"].shift()).abs()
    broken = s.loc[gap > 0.01, "coil"].unique()
    complete = s.groupby("coil")["stand_no"].agg(lambda x: sorted(x) == [1, 2, 3, 4, 5])
    ok = complete[complete].index.difference(broken)
    return df.assign(chain_ok=df["coil"].isin(ok))


def check(df: pd.DataFrame) -> None:
    per_coil = df.groupby("coil")["stand"].size()
    print(f"rows={len(df):,}  coils={df['coil'].nunique():,}")
    print("스탠드 5개가 아닌 코일:", per_coil[per_coil != 5].to_dict())

    # 앞 스탠드 출측 두께 = 다음 스탠드 입측 두께인지 확인
    s = df.sort_values(["coil", "stand_no"])
    prev_out = s.groupby("coil")["h_out"].shift()
    gap = (s["h_in"] - prev_out).abs().dropna()
    print(f"두께 연결 불일치(>0.01mm): {(gap > 0.01).sum()}건")

    print("음수/0 값:", int((df[["h_in", "h_out", "R", "W", "T_K", "F_N"]] <= 0).sum().sum()))
    print("h_out >= h_in:", int((df["h_out"] >= df["h_in"]).sum()))
    bad = df.loc[~df["chain_ok"]].groupby("grade")["coil"].nunique()
    print("chain_ok=False 코일:", bad.to_dict())


if __name__ == "__main__":
    df = add_chain_flag(add_features(load_raw()))
    check(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("saved:", OUT.relative_to(ROOT))
