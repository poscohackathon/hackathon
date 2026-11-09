"""탐색 분석: 주요 그래프(figures/*.png)와 요약표(outputs/*.csv).
실행: python src/eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# 스타일: 단일 계열은 파랑 하나, 강조는 주황, 격자·축은 흐리게
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

df = pd.read_csv(ROOT / "data/processed/rolling.csv")
STANDS = ["F1", "F2", "F3", "F4", "F5"]


def boxplot_by_stand(col, ylabel, title, fname, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df.loc[df.stand == s, col] for s in STANDS]
    bp = ax.boxplot(data, tick_labels=STANDS, widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=2))
    for b in bp["boxes"]:
        b.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=BLUE)
    for i, d in enumerate(data, start=1):
        ax.text(i + 0.3, d.median(), fmt.format(d.median()), va="center", fontsize=8, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    fig.savefig(FIG / fname)
    plt.close(fig)


# 1) 스탠드별 하중, 압하율, 온도 분포
boxplot_by_stand("F_MN", "압연 하중 (MN)", "스탠드별 압연 하중: 앞단(F1·F2)이 가장 크다", "01_force_by_stand.png")
boxplot_by_stand("reduction", "압하율", "스탠드별 압하율: 뒤로 갈수록 작아진다", "02_reduction_by_stand.png", "{:.2f}")
boxplot_by_stand("T_C", "소재 온도 (°C)", "스탠드별 소재 온도: 스탠드마다 약 25°C씩 떨어진다", "03_temp_by_stand.png", "{:.0f}")

# 2) 하중 vs 압하율 (스탠드별 small multiples, 밀도 hexbin)
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
for ax, s in zip(axes, STANDS):
    d = df[df.stand == s]
    ax.hexbin(d.reduction, d.F_MN, gridsize=30, cmap="Blues", mincnt=1, linewidths=0)
    r = np.corrcoef(d.reduction, d.F_MN)[0, 1]
    ax.set_title(f"{s}  (상관 {r:.2f})", fontsize=10, color=INK)
    ax.set_xlabel("압하율")
axes[0].set_ylabel("압연 하중 (MN)")
fig.suptitle("압하율이 클수록 하중이 커진다: 압하 배분이 곧 스탠드 부하를 결정", x=0.01, y=1.04, ha="left", color=INK)
fig.savefig(FIG / "04_force_vs_reduction.png")
plt.close(fig)

# 3) 강종별 평균 변형 저항 (하중 / (폭 × 접촉길이)): 강종이 '단단함'을 바꾼다
k = df.groupby("grade")["k_mean"].median().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(k.index, k.values, color=BLUE, height=0.6)
for y, v in enumerate(k.values):
    ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=8, color=INK)
ax.set_xlabel("평균 변형 저항 근사 (MPa, 중앙값)")
ax.set_title("강종별 변형 저항: 강종마다 같은 조건에서도 하중이 다르다", loc="left", color=INK, fontsize=12)
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "05_resistance_by_grade.png")
plt.close(fig)

# 4) 설비 이용률: 하중 / 스탠드별 과거 최대(상위 0.5%) — 최적화의 기준
cap = df.groupby("stand")["F_MN"].quantile(0.995)
df["util"] = df["F_MN"] / df["stand"].map(cap)
boxplot_by_stand("util", "설비 이용률 (하중 ÷ 과거 최대 하중)",
                 "스탠드별 설비 이용률: 평균은 비슷하지만 코일마다 편차가 크다", "06_utilization_by_stand.png", "{:.2f}")

# 코일별로 이용률이 가장 높은 스탠드 = 병목 스탠드
cw = df[df.chain_ok].pivot_table(index="coil", columns="stand", values="util")
bottleneck = cw.idxmax(axis=1).value_counts(normalize=True).reindex(STANDS).fillna(0)
fig, ax = plt.subplots(figsize=(7, 3.6))
colors = [ORANGE if v == bottleneck.max() else BLUE for v in bottleneck.values]
ax.bar(STANDS, bottleneck.values * 100, color=colors, width=0.55)
for x, v in enumerate(bottleneck.values):
    ax.text(x, v * 100 + 1, f"{v * 100:.0f}%", ha="center", fontsize=9, color=INK)
ax.set_ylabel("코일 비율 (%)")
ax.set_ylim(0, bottleneck.max() * 100 * 1.18)
ax.set_title("코일마다 병목 스탠드가 다르다: 고정 설정표보다 코일별 배분이 필요", loc="left", color=INK, fontsize=12)
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "07_bottleneck_stand.png")
plt.close(fig)

# 요약표
summary = df.groupby("stand").agg(
    rows=("F_MN", "size"),
    force_mean_MN=("F_MN", "mean"),
    force_p995_MN=("F_MN", lambda x: x.quantile(0.995)),
    reduction_mean=("reduction", "mean"),
    reduction_min=("reduction", "min"),
    reduction_max=("reduction", "max"),
    temp_mean_C=("T_C", "mean"),
    roll_radius_mean=("R", "mean"),
    util_mean=("util", "mean"),
).round(3)
grades = df.groupby("grade").agg(coils=("coil", "nunique"), force_mean_MN=("F_MN", "mean"),
                                 k_median_MPa=("k_mean", "median"), width_mean=("W", "mean")).round(2)
summary.to_csv(OUT / "eda_stand_summary.csv")
grades.to_csv(OUT / "eda_grade_summary.csv")
pd.set_option("display.width", 200)
print(summary, "\n")
print(grades, "\n")
print("병목 스탠드 비율:", (bottleneck * 100).round(1).to_dict())
print("최종 두께(F5 출측) 분포:", df[df.stand == "F5"].h_out.describe().round(2).to_dict())
print("saved figures:", sorted(p.name for p in FIG.glob("*.png")))
