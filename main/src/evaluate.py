"""하중 예측 모델 성능 평가: 물리식(논문 M3) / MLP / GBDT(단조 제약).

평가 방식: 처음 보는 코일 20% (코일 단위 분할, GroupShuffleSplit, random_state=42).
같은 코일의 F1~F5가 학습과 평가에 나뉘어 들어가지 않는다.
개발 과정과 다른 평가(코일 5-fold, 파라미터 가설 시험)는 ojs/01~05 노트북.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from data_utils import STANDS
from force_model import features, train
from optimize import physics_predictor


def split(df, test_size=0.2, seed=42):
    """코일 단위로 학습/평가를 나눈다."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                  .split(df, groups=df["Coil ID"]))
    return df.iloc[tr], df.iloc[te]


def train_mlp_predict(train_df, test_df, epochs=60, seed=0):
    """MLP (PyTorch, 128 → 64). 입력은 GBDT와 같은 변수를 표준화하고 강종은 원-핫으로."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(1)          # 작은 신경망은 스레드 1개가 가장 빠름
    torch.manual_seed(seed)
    Xtr, Xte = features(train_df), features(test_df)
    num = [c for c in Xtr.columns if c != "grade"]
    mu, sd = Xtr[num].mean(), Xtr[num].std()

    def to_tensor(X):
        onehot = pd.get_dummies(X["grade"]).astype(float)
        return torch.tensor(np.hstack([((X[num] - mu) / sd).to_numpy(), onehot.to_numpy()]), dtype=torch.float32)

    y_mean, y_std = train_df["force_MN"].mean(), train_df["force_MN"].std()
    y = torch.tensor(((train_df["force_MN"] - y_mean) / y_std).to_numpy(), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(len(num) + 10, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loader = DataLoader(TensorDataset(to_tensor(Xtr), y), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(net(xb).squeeze(1), yb).backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        return net(to_tensor(Xte)).squeeze(1).numpy() * y_std + y_mean


def more_reduction(rows, share=0.05):
    """출측 두께를 압하량의 5%만큼 더 얇게 (= 조금 더 누름). 단조성 점검용."""
    out = rows.copy()
    out["Exit gauge (mm)"] = rows["Exit gauge (mm)"] - share * (rows["Entry gauge (mm)"] - rows["Exit gauge (mm)"])
    return out


def evaluate(df):
    """세 모델을 같은 평가 코일로 비교한다. (성능표, 스탠드별 R², 예측값 dict, 평가 데이터)를 돌려준다."""
    train_df, test_df = split(df)
    y = test_df["force_MN"].to_numpy()

    gbdt = train(train_df, monotonic=True)
    preds = {
        "물리식 M3 (기준선)": np.asarray(physics_predictor(train_df)(test_df)),
        "MLP": train_mlp_predict(train_df, test_df),
        "GBDT (단조 제약)": gbdt.predict(features(test_df)),
    }
    p_more = gbdt.predict(features(more_reduction(test_df)))

    table = pd.DataFrame({
        name: {"R2": r2_score(y, p), "MAE_MN": mean_absolute_error(y, p),
               "MAPE_pct": np.mean(np.abs(p - y) / y) * 100}
        for name, p in preds.items()}).T
    table.loc["GBDT (단조 제약)", "monotonic_violation_pct"] = np.mean(p_more < preds["GBDT (단조 제약)"] - 1e-9) * 100

    stand = test_df["Stand ID"].to_numpy()
    stand_r2 = pd.DataFrame({
        name: {s: r2_score(y[stand == s], p[stand == s]) for s in STANDS} for name, p in preds.items()}).T
    return table, stand_r2, preds, test_df


def save_scatter(preds, test_df, path):
    """실제-예측 산점도: 물리식 vs GBDT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        if font in installed:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

    y = test_df["force_MN"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, name, color in zip(axes, ["물리식 M3 (기준선)", "GBDT (단조 제약)"], ["#898781", "#2a78d6"]):
        p = preds[name]
        ax.scatter(y, p, s=3, alpha=0.2, color=color, linewidths=0)
        ax.plot([0, 42], [0, 42], color="#52514e", linewidth=1, linestyle="--")
        ax.set_xlim(0, 42)
        ax.set_ylim(0, 42)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nR² {r2_score(y, p):.3f}, MAE {mean_absolute_error(y, p):.2f} MN")
        ax.set_xlabel("실제 하중 (MN)")
        ax.grid(color="#e1e0d9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("예측 하중 (MN)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
