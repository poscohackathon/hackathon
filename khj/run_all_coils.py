"""전체 코일에 최적화를 돌려 발표 숫자(전체 요약)와 시연 코일 표를 만든다.

optimize.py의 optimize()와 같은 규칙(같은 후보, seed 0)이라 앱에서 코일 하나를 돌린 결과와 숫자가 같다.
속도를 위해 코일 여러 개의 후보를 모아 모델을 한 번에 부른다.

실행: python3 khj/run_all_coils.py        (최종 GBDT 기준, 약 10~20분)
출력: khj/outputs/opt_all_coils.csv     코일별 현재·추천 배분, 하중, 이용률
      khj/outputs/demo_coils.csv        시연 후보 (아래 조건)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize import (N_STANDS, STANDS, balance_objective, coil_from_data,  # noqa: E402
                      final_gbdt_predictor, in_range, limits_from_data, load_data,
                      sample_candidates, schedules_to_rows)

OUT = Path(__file__).resolve().parent / "outputs"
BATCH = 40          # 한 번에 모델에 넣을 코일 수


def run(df, lim, predict_fn, n=3000, seed=0):
    coils = sorted(df["Coil ID"].unique())
    out, t = [], time.time()
    for i in range(0, len(coils), BATCH):
        jobs = []
        for cid in coils[i:i + BATCH]:
            coil = coil_from_data(cid, df)
            R = sample_candidates(coil, lim, n, seed)
            cur = np.asarray(coil.reductions)
            jobs.append((cid, coil, np.vstack([R, cur])))        # 마지막 줄 = 현재 배분
        rows = pd.concat([schedules_to_rows(c, R, lim) for _, c, R in jobs], ignore_index=True)
        F_all = np.asarray(predict_fn(rows), float).reshape(-1, N_STANDS)

        k = 0
        for cid, coil, R in jobs:
            F = F_all[k:k + len(R)]
            k += len(R)
            if len(R) == 1:                                       # 범위 안 후보 없음
                continue
            score = balance_objective(F[:-1], lim)
            b = int(np.argmin(score))
            u_opt, u_cur = F[b] / lim.cap, F[-1] / lim.cap
            actual = df.loc[df["Coil ID"] == cid].sort_values("Stand ID")["force_MN"].to_numpy()
            out.append(dict(
                coil=cid, grade=coil.grade, caster=coil.caster, width=round(coil.width),
                h0=round(coil.h0, 2), hf=round(coil.h_final, 2), n_cand=len(R) - 1,
                in_range=in_range(R[-1], lim),
                bott_cur=STANDS[u_cur.argmax()], bott_opt=STANDS[u_opt.argmax()],
                umax_cur=u_cur.max(), umax_opt=u_opt.max(),
                Fmax_cur=F[-1].max(), Fmax_opt=F[b].max(),
                act_err=np.abs(F[-1] - actual).max(),                  # 현재 배분 예측 vs 실측
                **{f"r{j+1}_cur": R[-1, j] for j in range(N_STANDS)},
                **{f"r{j+1}_opt": R[b, j] for j in range(N_STANDS)},
                **{f"F{j+1}_cur": F[-1, j] for j in range(N_STANDS)},
                **{f"F{j+1}_opt": F[b, j] for j in range(N_STANDS)},
            ))
        print(f"{min(i + BATCH, len(coils)):,}/{len(coils):,}코일  {time.time() - t:.0f}초", flush=True)
    o = pd.DataFrame(out)
    o["gain_pct"] = (1 - o.umax_opt / o.umax_cur) * 100
    return o


def summarize(o):
    g = o.gain_pct
    print(f"\n전체 {len(o):,}코일 (후보가 없는 코일 제외)")
    print(f"  최대 이용률 감소율 중앙값 {g.median():.1f}%, 평균 {g.mean():.1f}%, 최대 {g.max():.1f}%")
    for th in (5, 10):
        print(f"  {th}% 이상 줄어드는 코일 {(g >= th).sum():,}개 ({(g >= th).mean() * 100:.1f}%)")
    print(f"  최대 하중(MN)도 같이 줄어드는 코일 {(o.Fmax_opt < o.Fmax_cur).mean() * 100:.0f}%")


def demo_candidates(o):
    """시연 후보: 현재 배분이 과거 범위 안, 현재 예측이 실측과 1.5 MN 이내, 후보 200개 이상,
    현재 병목 F1, 최대 하중도 같이 줄어듦. 감소율 순."""
    c = o[o.in_range & (o.act_err < 1.5) & (o.n_cand >= 200)
          & (o.bott_cur == "F1") & (o.Fmax_opt < o.Fmax_cur)]
    return c.sort_values("gain_pct", ascending=False)


if __name__ == "__main__":
    df = load_data()
    lim = limits_from_data(df)
    o = run(df, lim, final_gbdt_predictor(df))
    OUT.mkdir(exist_ok=True)
    o.round(4).to_csv(OUT / "opt_all_coils.csv", index=False)
    summarize(o)
    d = demo_candidates(o)
    d.round(4).to_csv(OUT / "demo_coils.csv", index=False)
    cols = ["coil", "grade", "caster", "width", "h0", "hf", "bott_cur", "bott_opt",
            "umax_cur", "umax_opt", "gain_pct", "Fmax_cur", "Fmax_opt"]
    pd.set_option("display.width", 200)
    print("\n시연 후보 상위 10개\n", d[cols].head(10).round(3).to_string(index=False))
    print("\n지정 코일\n", o[o.coil.isin([2744, 2599, 2086, 6175, 6920])][cols].round(3).to_string(index=False))
