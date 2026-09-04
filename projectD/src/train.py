"""
학습 스크립트 — 대시보드가 쓸 모델을 만듭니다
=============================================
    python src/train.py

원칙 (이 프로젝트의 핵심 규칙 3가지)
  1) 목표변수는 '30분 내 고장' (감지가 아니라 예지)
  2) split은 반드시 시간순
  3) 임계값은 0.5가 아니라 비용 최소점
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)  # 학습 모델과 평가 지표 함수들

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F  # noqa: E402  # features.py 전체를 F로 불러옴
from clean import run_pipeline  # noqa: E402  # clean.py에서 run_pipline 함수만 가져옴


# 상수 정의
HORIZON = 30  # 30분 안에 고장나는가 예측
COST_FN = 8_000_000  # 실제 고장인데 놓친 경우의 비용은 800만원
COST_FP = 300_000  # 정상인데 잘못 경고한 경우의 비용은 30만원
SEED = 42  # 재현성 시드


# CSV 모아 읽기
def load_history() -> pd.DataFrame:
    files = sorted((ROOT / "data" / "history").glob("*.csv"))
    if not files:
        raise SystemExit(
            "data/history/*.csv 가 없습니다. collector.py를 먼저 돌리세요."
        )
    return pd.concat(
        [pd.read_csv(f) for f in files], ignore_index=True
    )  # 파일 전부 읽어 하나로 합침


def main() -> int:
    raw = load_history()  # CSV 다 읽어 합친 원본
    print(
        f"원본 {len(raw):,}행 ({len(list((ROOT / 'data' / 'history').glob('*.csv')))}개 파일)"
    )  # 몇 행/몇 파일인지 출력

    clean, log, rep = run_pipeline(
        raw, verbose=True
    )  # log, rep는 변수에 담기만 하고 안 씀. 어차피 verbose=True로 콘솔에 단계별 표 출력
    clean = clean[
        ~clean["is_gap"].astype(bool)
    ].copy()  # is_gap=True인 행(빠진 시각들) 걸러내고 ~로 뒤집어 "gap이 아닌 행만 남김"

    d = F.make_horizon_label(clean, horizon=HORIZON)  # 정제된 데이터에 features.py 적용
    d = F.build(d, windows=(10, 30, 60), shift_one=True)
    feat = [c for c in F.feature_columns(d) if c not in ("y", "machine_failure")]
    d = (
        d.dropna(subset=feat + ["y"]).sort_values("ts").reset_index(drop=True)
    )  # 그동안 일부러 안 채우고 남겨둔 모든 "NaN"이 정리되는 지점. feat(피처 컬럼들)이나 y(라벨) 중 하나라도 NaN인 행은 통째로 버림.
    # 물리범위 밖이라 NaN이 됐는데 못 채워진 값들
    # 긴 결측이라 안 채워진 값들
    # add_rolling()의 shift(1) 때문에 각 설비의 맨 처음 몇 개 행은 window가 아직 안 찼으니 자동으로 NaN
    # make_horizon_label()의 shift(-1) 때문에 각 설비의 맨 끝 부분도 미래 window가 부족해서 NaN
    if len(d) < 500 or d["y"].nunique() < 2:
        raise SystemExit(
            f"학습에 쓸 데이터가 부족합니다({len(d)}행). 수집을 더 하세요."
        )  # 데이터 부족 안전장치 -> 전체 행 수가 500보다 작으면 학습 의미 x, 라벨 전부 0이거나 1이면 분류모델 학습 x

    # ------------ 여기까지 데이터 준비 단계 ------------

    X = d[feat].values  # 피처 컬럼들만 뽑아서 numpy 배열로
    y = d["y"].values.astype(int)  # 라벨 int 변환
    cut = int(len(d) * 0.75)  # ★ 시간순 split -> 앞 75%는 훈련용, 뒤 25% 테스트용
    print(
        f"\ntrain {cut:,} / test {len(d) - cut:,} | "
        f"train 양성률 {y[:cut].mean() * 100:.2f}% / test {y[cut:].mean() * 100:.2f}%"
    )

    mdl = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2
    )  # 트리 300개 만들어서 앙살블/트리 생성 시 무작위성 고정해서 재현 가능하게/병렬로 학습/샘플 2개 이상은 무조건 가져야 함(과적합 방지)
    mdl.fit(X[:cut], y[:cut])  # 학습 수행
    p = mdl.predict_proba(X[cut:])[:, 1]  # 0,1에 속할 확률 중 1인 고장일 확률만 뽑음
    yt = y[cut:]  # 테스트 구간 정답 라벨

    # 비용 최소 임계값
    ths = np.linspace(0.01, 0.99, 197)
    costs = []
    for t in ths:
        tn, fp, fn, tp = confusion_matrix(
            yt, (p >= t).astype(int), labels=[0, 1]
        ).ravel()
        costs.append(fn * COST_FN + fp * COST_FP)
    i = int(np.argmin(costs))
    th = float(ths[i])

    metrics = {
        "n_rows": int(len(d)),
        "n_features": len(feat),
        "horizon_min": HORIZON,
        "train_end": str(d["ts"].iloc[cut - 1]),
        "test_start": str(d["ts"].iloc[cut]),
        "positive_rate_test": round(float(yt.mean()), 4),
        "roc_auc": round(float(roc_auc_score(yt, p)), 4),
        "pr_auc": round(float(average_precision_score(yt, p)), 4),
        "threshold": round(th, 3),
        "precision": round(
            float(precision_score(yt, (p >= th).astype(int), zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(yt, (p >= th).astype(int), zero_division=0)), 4
        ),
        "f1": round(float(f1_score(yt, (p >= th).astype(int), zero_division=0)), 4),
        "cost_at_threshold": int(costs[i]),
        "cost_at_0.5": int(costs[int(np.argmin(np.abs(ths - 0.5)))]),
    }
    print("\n[성능]")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")

    imp = pd.Series(mdl.feature_importances_, index=feat).sort_values(ascending=False)
    print("\n[변수 중요도 상위 10]")
    print(imp.head(10).round(4).to_string())

    outdir = ROOT / "models"
    outdir.mkdir(exist_ok=True)
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    imp.head(30).to_csv(outdir / "feature_importance.csv", header=["importance"])
    pd.DataFrame(
        {
            "ts": d["ts"].iloc[cut:].values,
            "machine_id": d["machine_id"].iloc[cut:].values,
            "y": yt,
            "prob": p,
        }
    ).to_csv(outdir / "test_predictions.csv", index=False)
    print(f"\n저장: models/metrics.json, feature_importance.csv, test_predictions.csv")
    print("★ 모델 파일(.pkl)은 저장하지 않습니다 — sklearn 버전이 다르면 못 읽습니다.")
    print(
        "  대시보드는 예측 결과 CSV를 읽습니다. 재현은 이 스크립트를 다시 돌리면 됩니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
