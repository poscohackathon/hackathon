"""AI 기반 열연 생산 시뮬레이션 및 최적 운영 의사결정 시스템 — 전체 실행.

실행 (저장소 맨 위 폴더에서):
    pip install -r main/requirements.txt
    python main/main.py          # 약 2분

하는 일
    1. 데이터 정리      : main/raw_data/metals.csv → 스탠드 누락·두께 불연속·폭 변동 코일 제외
    2. 하중 예측 평가   : 물리식(논문 M3) / MLP / GBDT(단조 제약)를 처음 보는 코일 20%로 비교
    3. 최종 모델 학습   : 정리 데이터 전체로 GBDT(단조 제약) 학습
    4. 압하 배분 최적화 : 시연 코일(2744, 2599) + 무작위 200코일 요약

결과는 main/outputs/에 저장한다.
    model_performance.csv, stand_r2.csv, actual_vs_pred.png,
    demo_coils.csv, optimization_200coils.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.dont_write_bytecode = True

from evaluate import evaluate, save_scatter          # noqa: E402
from force_model import make_predict_fn, train       # noqa: E402
from optimize import coil_from_data, limits_from_data, load_data, optimize  # noqa: E402

OUT = HERE / "outputs"
STANDS = ["F1", "F2", "F3", "F4", "F5"]
DEMO_COILS = [2744, 2599]


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)

    # 1. 데이터
    df = load_data()
    print(f"[1] 데이터: {len(df):,}행, 코일 {df['Coil ID'].nunique():,}개")

    # 2. 하중 예측 평가
    table, stand_r2, preds, test_df = evaluate(df)
    table.to_csv(OUT / "model_performance.csv", encoding="utf-8-sig")
    stand_r2.to_csv(OUT / "stand_r2.csv", encoding="utf-8-sig")
    save_scatter(preds, test_df, OUT / "actual_vs_pred.png")
    print("\n[2] 하중 예측 성능 (처음 보는 코일 20%)")
    print(table.round(3).to_string())
    print("\n스탠드별 R²")
    print(stand_r2.round(2).to_string())

    # 3. 최종 모델
    predict_fn = make_predict_fn(train(df, monotonic=True))
    lim = limits_from_data(df)
    print("\n[3] 최종 모델 학습 완료 (정리 데이터 전체, GBDT 단조 제약)")

    # 4-1. 시연 코일
    rows = []
    print("\n[4] 시연 코일 최적화")
    for cid in DEMO_COILS:
        coil = coil_from_data(cid, df)
        res = optimize(coil, predict_fn, lim)
        cur, best = res["current"], res["best"]
        actual = df[df["Coil ID"] == cid].sort_values("Stand ID")["force_MN"].to_numpy()
        for i, s in enumerate(STANDS):
            rows.append({"coil": cid, "grade": coil.grade, "stand": s, "actual_force_MN": actual[i],
                         "current_reduction": cur["reductions"][i], "current_force_MN": cur["forces_MN"][i],
                         "current_util": cur["util"][i], "best_reduction": best["reductions"][i],
                         "best_force_MN": best["forces_MN"][i], "best_util": best["util"][i]})
        print(f"  코일 {cid}: 최대 이용률 {cur['util'].max():.3f} → {best['util'].max():.3f} "
              f"(−{res['util_max_drop_pct']:.1f}%), 최대 하중 {cur['forces_MN'].max():.1f} → "
              f"{best['forces_MN'].max():.1f} MN, 병목 {cur['bottleneck']} → {best['bottleneck']}")
    pd.DataFrame(rows).to_csv(OUT / "demo_coils.csv", index=False, encoding="utf-8-sig")

    # 4-2. 무작위 200코일 요약
    ids = df["Coil ID"].drop_duplicates().sample(200, random_state=0).tolist()
    rows, skipped = [], 0
    for cid in ids:
        try:
            res = optimize(coil_from_data(cid, df), predict_fn, lim)
        except ValueError:          # 과거 운영 범위 안에서 목표 두께를 맞출 수 없는 코일
            skipped += 1
            continue
        cur, best = res["current"], res["best"]
        rows.append({"coil": cid, "bottleneck_current": cur["bottleneck"], "bottleneck_best": best["bottleneck"],
                     "util_max_current": cur["util"].max(), "util_max_best": best["util"].max(),
                     "util_max_drop_pct": res["util_max_drop_pct"],
                     "force_max_current_MN": cur["forces_MN"].max(), "force_max_best_MN": best["forces_MN"].max()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "optimization_200coils.csv", index=False, encoding="utf-8-sig")
    print(f"\n  무작위 200코일 (범위 밖 {skipped}개 제외, {len(r)}개):")
    print(f"  최대 이용률 감소 중앙값 {r['util_max_drop_pct'].median():.1f}%, 평균 {r['util_max_drop_pct'].mean():.1f}%, "
          f"5% 이상 개선 {np.mean(r['util_max_drop_pct'] >= 5) * 100:.1f}%, 최대 {r['util_max_drop_pct'].max():.1f}%")
    print(f"  최대 하중(MN)이 늘어난 코일 {np.mean(r['force_max_best_MN'] > r['force_max_current_MN']) * 100:.0f}%")

    print(f"\n완료 ({time.time() - start:.0f}초). 결과: {OUT}")


if __name__ == "__main__":
    main()
