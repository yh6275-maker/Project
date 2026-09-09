"""
A3. ★★★ 시간 누수 (temporal leakage) 실측 대조
================================================
이 프로젝트의 하이라이트입니다.

먼저 문제를 제대로 정의합니다.
  감지(detection) : "지금 고장인가?"      → 이미 늦었습니다
  예지(prediction): "30분 안에 고장날까?"  → 이게 예지보전입니다

★ 저는 처음에 감지 문제로 풀었다가, 시간 누수가 거의 안 나타나서 원인을 찾았습니다.
  그 과정을 A3-6에 그대로 남겼습니다. 실패 기록도 자료입니다.
"""

from _common import ROOT, RAW, rule, save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (
    NearestNeighbors,
)  # 가장 가까운 점을 찾아줌 -> test행이 train행과 얼마나 가까운지 거리
from sklearn.preprocessing import StandardScaler

import features as F

pd.set_option("display.width", 170)
SEED = 42
HORIZON = 30  # 분
rng = np.random.default_rng(SEED)


# train.py에서의 학습 로직
def evaluate(Xtr, ytr, Xte, yte, seed=SEED):
    m = RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=2
    )
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return (
        {
            "ROC-AUC": round(roc_auc_score(yte, p), 4),
            "PR-AUC": round(average_precision_score(yte, p), 4),
            "F1@0.5": round(f1_score(yte, (p >= 0.5).astype(int), zero_division=0), 4),
            "양성률": round(float(np.mean(yte)), 4),
        },
        m,
        p,
    )


# ======================================================================
rule("A3-1. 문제 정의 — 감지가 아니라 예지")
path = ROOT / "data" / "clean_sim.parquet"
if not path.exists():
    path = ROOT / "data" / "clean_sim.csv"
sim = (
    pd.read_parquet(path)
    if path.suffix == ".parquet"
    else pd.read_csv(path, parse_dates=["ts"])
)
sim["ts"] = pd.to_datetime(sim["ts"])
sim = sim[~sim["is_gap"].astype(bool)].copy()  # is_gap 제거
print("정제된 시뮬레이터 데이터:", sim.shape)
print("기간:", sim["ts"].min(), "~", sim["ts"].max())
print(
    f"'지금 고장' 비율(감지 라벨) : {sim['machine_failure'].mean() * 100:.3f}%"
)  # 순간 고장 여부 자체의 비율

sim = F.make_horizon_label(sim, horizon=HORIZON)
print(f"'{HORIZON}분 내 고장' 비율(예지 라벨): {sim['y'].mean() * 100:.2f}%")
print(f"""
★ 예지 라벨은 미래 {HORIZON}분 창의 최댓값입니다.
  → t행과 t+1행은 창이 {HORIZON - 1}/{HORIZON} 겹칩니다. 라벨이 거의 같습니다.
  이 '겹침'이 뒤에서 누수의 핵심 원인이 됩니다.""")

sim = F.build(sim, windows=(10, 30, 60), shift_one=True)
feat = [c for c in F.feature_columns(sim) if c not in ("y", "machine_failure")]
d = sim.dropna(subset=feat + ["y"]).sort_values("ts").reset_index(drop=True)
print(
    f"\n피처 {len(feat)}개 | 결측 제거 후 {len(d):,}행 "
    f"| 양성 {int(d['y'].sum()):,}건 ({d['y'].mean() * 100:.2f}%)"
)

X = d[feat].values
y = d["y"].values.astype(int)
ts = d["ts"].values

# ----------------------------------------------------------------------
rule("A3-2. ★ 자기상관 확인 — 이게 누수의 원인입니다")
one = d[d["machine_id"] == "CNC-01"]
print("[CNC-01 시차별 자기상관]")
# s.autocorr(k) : pandas의 자기상관 함수
# -> 이 시계열이 k만큼 시간이 지난 자기 자신과 얼마나 비슷한가를 -1~1 사이 숫자로 알려줌
ac = pd.DataFrame(
    {
        "토크": [
            round(one["torque_nm"].autocorr(k), 4) for k in [1, 5, 10, 30, 60, 120]
        ],
        "진동": [
            round(one["vibration_mms"].autocorr(k), 4) for k in [1, 5, 10, 30, 60, 120]
        ],
        "예지라벨 y": [round(one["y"].autocorr(k), 4) for k in [1, 5, 10, 30, 60, 120]],
    },
    index=[f"lag {k}분" for k in [1, 5, 10, 30, 60, 120]],
)
print(ac.to_string())
print("""
★ 라벨 y의 1분 자기상관이 0.97을 넘습니다.
  t행과 t+1행은 피처도 거의 같고 라벨도 거의 같습니다 = 사실상 같은 행.
  랜덤 split은 이 쌍을 train/test로 갈라 놓습니다.
  → 모델은 '학습'이 아니라 '옆자리 답 보기'를 하게 됩니다.""")

# ----------------------------------------------------------------------
rule("A3-3. ★★★ 랜덤 split vs 시간순 split — 실측 대조")
n = len(d)
cut = int(n * 0.75)
itr, ite = train_test_split(np.arange(n), test_size=0.25, random_state=SEED, stratify=y)
# 인덱스를 랜덤하게 25/75%로 나눔 -> 잘못된 방법 재현
res_rand, m_rand, p_rand = evaluate(X[itr], y[itr], X[ite], y[ite])
itr2, ite2 = np.arange(cut), np.arange(cut, n)
# 시간순으로 앞 75, 뒤 25 나누는 방법
res_time, m_time, p_time = evaluate(X[itr2], y[itr2], X[ite2], y[ite2])
# itr, ite : 랜덤하게 섞인 인덱스 / itr2, ite2 : 시간순 인덱스
# 나누는 방법 하나만 바꾼 비교

cmp = pd.DataFrame(
    [
        dict(split="랜덤 split (잘못)", **res_rand),
        dict(split="시간순 split (올바름)", **res_time),
    ]
)
print(cmp.to_string(index=False))

# 하락폭 계산 - 랜덤 split 대비 시간순 split이 몇 % 낮은지
d_pr = (res_rand["PR-AUC"] - res_time["PR-AUC"]) / res_rand["PR-AUC"] * 100
d_f1 = (res_rand["F1@0.5"] - res_time["F1@0.5"]) / res_rand["F1@0.5"] * 100
d_roc = (res_rand["ROC-AUC"] - res_time["ROC-AUC"]) / res_rand["ROC-AUC"] * 100
print(f"""
★★★ 같은 데이터, 같은 모델, 같은 하이퍼파라미터입니다. split만 바꿨습니다.

     ROC-AUC  {res_rand["ROC-AUC"]:.4f} → {res_time["ROC-AUC"]:.4f}   ({d_roc:5.1f}% 하락)
     PR-AUC   {res_rand["PR-AUC"]:.4f} → {res_time["PR-AUC"]:.4f}   ({d_pr:5.1f}% 하락)
     F1@0.5   {res_rand["F1@0.5"]:.4f} → {res_time["F1@0.5"]:.4f}   ({d_f1:5.1f}% 하락)

  랜덤 split 숫자를 포트폴리오에 쓰면 실제 운영 성능을 크게 부풀리는 겁니다.
  "PR-AUC 0.98 달성!"이라고 썼는데 실제로는 0.76입니다.""")
# 시간 누수가 났을 때는 PR-AUC가 0.9820이다.
# 그러나 시간순대로 수행하면 0.7605 !! -> 시간 누수가 일어나 성능이 부풀려졌다는 증거

print("\n[구간 정보]")
print(f"  train {pd.Timestamp(ts[0])} ~ {pd.Timestamp(ts[cut - 1])}  ({cut:,}행)")
print(f"  test  {pd.Timestamp(ts[cut])} ~ {pd.Timestamp(ts[-1])}  ({n - cut:,}행)")
print(
    f"  train 양성률 {y[:cut].mean() * 100:.2f}% / test 양성률 {y[cut:].mean() * 100:.2f}%"
)

# ----------------------------------------------------------------------
rule("A3-4. 메커니즘 확인 — test 행과 가장 비슷한 train 행의 거리")
sc = StandardScaler().fit(X)  # 모든 피처를 평균 0, 표준편차 1로 맞춤
Xs = sc.transform(X)


# 최근접 이웃 거리 계산 함수
def nn_dist(tr_idx, te_idx, sample=3000):
    tr = Xs[tr_idx]
    te = Xs[te_idx]
    if len(te) > sample:
        te = te[rng.choice(len(te), sample, replace=False)]
    nn = NearestNeighbors(n_neighbors=1).fit(tr)  # 가장 가까운 1개 이웃을 찾아주는 도구
    dist, _ = nn.kneighbors(
        te
    )  # test의 각 행에 대해, train에서 가장 가까운 점까지의 거리와 그 점의 위치(인덱스) 돌려줌
    return dist.ravel()


# 랜덤 split vs 시간순 split 거리 비교(중앙값, 5%, 95% 백분위수)
dr = nn_dist(itr, ite)
dt = nn_dist(itr2, ite2)
print(
    pd.DataFrame(
        {
            "split": ["랜덤", "시간순"],
            "최근접거리 중앙값": [
                round(float(np.median(dr)), 4),
                round(float(np.median(dt)), 4),
            ],
            "p05": [
                round(float(np.percentile(dr, 5)), 4),
                round(float(np.percentile(dt, 5)), 4),
            ],
            "p95": [
                round(float(np.percentile(dr, 95)), 4),
                round(float(np.percentile(dt, 95)), 4),
            ],
        }
    ).to_string(index=False)
)

# 시간 간격도 함께 봅니다 (이게 더 직관적입니다)
tr_t = ts[itr].astype("datetime64[m]").astype(np.int64)
te_t = ts[ite].astype("datetime64[m]").astype(np.int64)
nn_t = NearestNeighbors(n_neighbors=1).fit(tr_t.reshape(-1, 1))
gap_rand = nn_t.kneighbors(te_t.reshape(-1, 1))[0].ravel()
tr_t2 = ts[itr2].astype("datetime64[m]").astype(np.int64)
te_t2 = ts[ite2].astype("datetime64[m]").astype(np.int64)
nn_t2 = NearestNeighbors(n_neighbors=1).fit(tr_t2.reshape(-1, 1))
gap_time = nn_t2.kneighbors(te_t2.reshape(-1, 1))[0].ravel()
print(f"""
[test 행에서 가장 가까운 train 행까지의 '시간' 거리]
  랜덤 split  : 중앙값 {np.median(gap_rand):.1f}분   (최대 {gap_rand.max():.0f}분)
  시간순 split: 중앙값 {np.median(gap_time):.1f}분  (최대 {gap_time.max():.0f}분)

★ 랜덤 split에서는 test 행 바로 옆(0~1분)에 train 행이 있습니다.
  예지 라벨의 미래 창이 {HORIZON}분이니, 창이 통째로 겹칩니다.
  정답을 이미 본 것과 다를 게 없습니다.""")

# ----------------------------------------------------------------------
rule("A3-5. 그럼 얼마나 떼어놓아야 하나 — gap(purging) 실험")
gap_rows = []
for gap_min in [0, 30, 60, 180, 720]:
    te_start = cut + gap_min * 3  # 설비 3대 × 분
    if te_start >= n - 500:
        continue
    r, _, _ = evaluate(X[:cut], y[:cut], X[te_start:], y[te_start:])
    gap_rows.append(dict(**{"train~test 간격(분)": gap_min}, **r))
print(pd.DataFrame(gap_rows).to_string(index=False))
print(f"""
★ 시간순 split은 간격을 벌려도 성능이 크게 변하지 않습니다({res_time["PR-AUC"]:.3f} 근처).
  즉 시간순 split은 이미 '정직한' 상태입니다.
  반대로 랜덤 split({res_rand["PR-AUC"]:.3f})은 어떤 gap으로도 설명되지 않는 이상치입니다.
  ★ 라벨 창이 겹치는 문제까지 막으려면 train 끝에서 {HORIZON}분을 잘라내는
    'purging'을 씁니다(금융 쪽에서 온 기법입니다).""")

# ----------------------------------------------------------------------
rule("A3-6. ★ 제가 처음에 틀렸던 것 — 감지 문제로 풀면 누수가 안 보입니다")
y_now = d["machine_failure"].values.astype(
    int
)  # 예지 라벨 y가 아니라, 지금 이 순간 고장인가인 원래의 감지 라벨
r_now_rand, _, _ = evaluate(X[itr], y_now[itr], X[ite], y_now[ite])
r_now_time, _, _ = evaluate(X[:cut], y_now[:cut], X[cut:], y_now[cut:])
# 랜덤 인덱스, 시간순 인덱스 그대로 재사용하되, 목표변수만 y(예지)에서 y_now(감지)로 바꿔서 다시 평가
print("[목표변수 = '지금 고장인가' (감지)]")
print(
    pd.DataFrame(
        [
            dict(split="랜덤 split", **r_now_rand),
            dict(split="시간순 split", **r_now_time),
        ]
    ).to_string(index=False)
)
print("\n[목표변수 = '30분 내 고장날까' (예지)]")
print(cmp.to_string(index=False))
print(f"""
★★ 감지 문제에서는 두 split 차이가 거의 없습니다
   (PR-AUC {r_now_rand["PR-AUC"]:.4f} vs {r_now_time["PR-AUC"]:.4f}).
   이유: '지금 고장'은 지금 센서값만 보면 알 수 있는 결정론적 규칙이라
   외울 필요 없이 규칙만 배우면 되고, 그 규칙은 시간이 지나도 안 변합니다.

   예지 문제로 바꾸는 순간 차이가 벌어집니다({d_pr:.0f}% 하락).
   미래를 맞히려면 '패턴'이 필요한데, 랜덤 split에서는 그냥 옆자리를 베끼면 되니까요.

   ★ 교훈: 누수는 '데이터가 시계열이냐'가 아니라
     '문제 정의가 무엇이냐'에 따라 생깁니다. 이걸 구분해서 말하세요.""")
# 감지 문제 - 랜덤이든 시간순이든 상관 x
#       train을 랜덤으로 섞어서 뽑든 시간순으로 앞부분만 쓰든 결국 같은 물리 규칙을 배우게 되고, 그 규칙은 test 구간 어디에 있든 잘 맞음
#       따라서 랜덤 split과 시간순 split 성능 차이가 거의 없음
#
# 예지 문제 - 랜덤 split이면 치팅!
#       30분 뒤에 고장날지를 알려주는 물리 공식은 없음. 불확실한 미래를 추론해야 하는 문제.
#       09:00분 행의 라벨은 "09:00~09:30 사이에 고장이 있나"인데, 이 구간이 09:01분 라벨 구간과 29분이나 겹침.
#       따라서 09:00분과 09:01분의 라벨은 사실상 거의 항상 같음
#       설비 피처값도 거의 동일하기 때문에 09:00분의 답을 베껴버림
#       "그러나 시간순 split에서는 통하지 않음!"
#       train에 있는 건 며칠 전 데이터뿐이라 방금 본 거의 똑같은 상황을 참고할 수 없다(베낄 수 없다).
#       진짜로 "패턴"을 배워야만 test를 맞힐 수 있음

# 이게 두 문제의 split 민감도가 다른 이유이다 !!!


# ----------------------------------------------------------------------
rule("A3-7. 또 다른 반전 — AI4I(독립 스냅샷)에서는 랜덤 split도 괜찮습니다")
ai = pd.read_csv(RAW / "ai4i" / "ai4i2020.csv").rename(
    columns={
        "Air temperature [K]": "air_temp_k",
        "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rot_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "failure",
        "Type": "type",
    }
)
NUM = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]
ai = ai.sort_values("UDI").reset_index(drop=True)
print("[AI4I 시차별 자기상관 — UDI 순서 기준]")
print(
    pd.Series(
        {f"lag {k}": round(ai["torque_nm"].autocorr(k), 4) for k in [1, 5, 10, 30, 60]}
    ).to_string()
)
# AI4I는 애초에 시각(ts) 컬럼이 없음. UDI(순번)만 있음 -> 자기상관 0에 가깝게 나옴
# 시뮬레이터처럼 연속된 시간의 흐름이 아님

Xa = pd.get_dummies(ai[NUM + ["type"]], columns=["type"], drop_first=True).values
Xa = np.hstack(
    [
        Xa,
        (ai["process_temp_k"] - ai["air_temp_k"]).values[:, None],
        (ai["torque_nm"] * ai["rot_speed_rpm"] * 2 * np.pi / 60).values[:, None],
        (ai["tool_wear_min"] * ai["torque_nm"]).values[:, None],
    ]
)
ya = ai["failure"].values
na = len(ai)
cuta = int(na * 0.75)
ia_tr, ia_te = train_test_split(
    np.arange(na), test_size=0.25, random_state=SEED, stratify=ya
)
ra, _, _ = evaluate(Xa[ia_tr], ya[ia_tr], Xa[ia_te], ya[ia_te])
rb, _, _ = evaluate(Xa[:cuta], ya[:cuta], Xa[cuta:], ya[cuta:])
# "랜덤 split"(ia_tr, ia_te)과 "UDI 순서 split"([:cuta], [cuta:]) 두 가지를 다 준비
print(
    "\n"
    + pd.DataFrame(
        [dict(split="랜덤 split", **ra), dict(split="UDI 순서 split", **rb)]
    ).to_string(index=False)
)
print("""
★ AI4I는 lag 1 자기상관이 0.005입니다. 연속 로그가 아니라 독립 스냅샷이라서요.
  그래서 랜덤 split이 미래를 흘리지 않습니다. PR-AUC 차이는 주로
  두 구간의 고장률이 다른 데서 옵니다(test 고장률 3.4% vs 2.2%).

  ★★ 정확한 원칙: "시계열이면 무조건 시간순 split"이 아니라
     "행 사이에 시간적 의존성이 있거나, 라벨 창이 겹치면 시간순 split".
     이 차이를 설명하면 외운 사람이 아니라 이해한 사람으로 보입니다.""")

# ----------------------------------------------------------------------
rule("A3-8. 전처리 누수 — split 전에 fit 하면")
sc_all = StandardScaler().fit(X)  # 잘못된 방법 - train과 test를 합친 전체 데이터
Xall = sc_all.transform(X)
r_leak, _, _ = evaluate(Xall[:cut], y[:cut], Xall[cut:], y[cut:])
sc_tr = StandardScaler().fit(
    X[:cut]
)  # 올바른 방법 - train 데이터만 가지고 평균, 표준편차 계산
r_ok, _, _ = evaluate(
    sc_tr.transform(X[:cut]), y[:cut], sc_tr.transform(X[cut:]), y[cut:]
)
print(
    pd.DataFrame(
        [
            dict(방식="전체로 스케일러 fit (누수)", **r_leak),
            dict(방식="train으로만 fit (올바름)", **r_ok),
        ]
    ).to_string(index=False)
)
print("""
★ 트리 모델은 스케일에 둔감해서 차이가 거의 없습니다. 솔직히 그렇게 씁니다.
  하지만 결측 대체값·인코딩 카테고리·이상치 임계값을 전체로 계산하면
  같은 방식으로 test 정보가 새어 들어갑니다.
  → 습관적으로 Pipeline 안에 넣어 train fold에서만 fit 되게 하세요.""")
# ex) 평균으로 결측치를 채운다고 할 때 그 평균을 train+test 전체로 계산하면 똑같이 시간 누수가 생긴다
# Pipeline을 쓰면 .fit(Xtr, ytr)을 호출할 때 자동으로 스케일러도 train 데이터로만 fit되고,
# 그 학습된 스케일러가 나중에 test에 적용될 때는 transform만 하도록 설계돼 있음


# ----------------------------------------------------------------------
rule("A3-9. 그림 & 저장")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
lbl = ["랜덤\nsplit", "시간순\nsplit"]
for ax, key in zip(axes[:2], ["PR-AUC", "F1@0.5"]):
    v = [res_rand[key], res_time[key]]
    b = ax.bar(lbl, v, color=["#c53030", "#2b6cb0"], width=0.55)
    ax.bar_label(b, fmt="%.3f", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{key} — 30분 내 고장 예지")
# ★ 시간 거리는 두 분포의 스케일이 1000배 달라 히스토그램이 안 보입니다.
#   → 분위수를 로그 축 막대로 그립니다.
qs = [50, 75, 90, 100]
r_v = [np.percentile(gap_rand, q) for q in qs]
t_v = [np.percentile(gap_time, q) for q in qs]
xx = np.arange(len(qs))
axes[2].bar(xx - 0.2, np.maximum(r_v, 0.5), 0.4, color="#c53030", label="랜덤 split")
axes[2].bar(xx + 0.2, np.maximum(t_v, 0.5), 0.4, color="#2b6cb0", label="시간순 split")
axes[2].axhline(HORIZON, color="k", ls="--", lw=1.2, label=f"라벨 창 {HORIZON}분")
axes[2].set_yscale("log")
axes[2].set_xticks(xx)
axes[2].set_xticklabels([f"p{q}" if q < 100 else "최대" for q in qs])
axes[2].set_ylabel("분 (로그 눈금)")
axes[2].set_ylim(0.35, 2.5e4)
axes[2].set_title("test→최근접 train 시간 거리")
for i, (a, b) in enumerate(zip(r_v, t_v)):
    axes[2].text(i - 0.2, max(a, 0.5), f"{a:.0f}", ha="center", va="bottom", fontsize=7)
    axes[2].text(i + 0.2, max(b, 0.5), f"{b:.0f}", ha="center", va="bottom", fontsize=7)
axes[2].legend(fontsize=7, loc="center left", framealpha=0.95)
fig.tight_layout()
save(fig, "leakage")

out = pd.DataFrame([dict(split="랜덤", **res_rand), dict(split="시간순", **res_time)])
out.to_csv(ROOT / "data" / "leakage_result.csv", index=False)
d[["ts", "machine_id", "y", "machine_failure"]].assign(prob_time=np.nan).iloc[
    cut:
].assign(prob_time=p_time).to_csv(ROOT / "data" / "sim_test_pred.csv", index=False)
print("저장: data/leakage_result.csv, data/sim_test_pred.csv")


# A3-1: 문제 정의 → 예지 라벨은 창이 29/30 겹친다는 걸 이론적으로 예고
# A3-2: 자기상관 → y의 lag1 자기상관 0.97로 실측 증명
# A3-3: ★★★ 랜덤 vs 시간순 → PR-AUC 22.6% 하락, 헤드라인 숫자 실측
# A3-4: 최근접 거리 → 랜덤split은 test 옆에 train이 1분 거리, 메커니즘 규명
# A3-5: gap 실험 → 시간순은 gap 늘려도 안정적, purging 개념 소개
# A3-6: ★ 감지 vs 예지 → 감지 라벨은 split 무관, 예지만 크게 갈림 (저자의 실패담)
# A3-7: AI4I 검증 → 독립 스냅샷은 자기상관 0.005, 랜덤 split도 안전
# A3-8: 전처리 누수 → 스케일러 fit 시점 문제, Pipeline 권장
# A3-9: 그림·저장 → 시각적 증거 + 후속 분석용 CSV
