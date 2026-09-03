"""
A8. 대시보드가 보여주는 데이터 품질 지표
========================================
Streamlit '데이터 품질' 탭의 숫자를 그대로 계산합니다.
앱과 같은 로직을 쓰므로, 앱 화면의 숫자와 일치해야 합니다(교차 검증).
"""
from _common import ROOT, rule
import pandas as pd

pd.set_option("display.width", 160)

rule("A8-1. 수집 이력 로드")
files = sorted((ROOT / "data" / "history").glob("*.csv"))
raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
raw["ts"] = pd.to_datetime(raw["ts"], errors="coerce")
NUM = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm",
       "tool_wear_min", "vibration_mms", "current_a", "humidity_pct"]
for c in NUM:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw.dropna(subset=["ts"]).drop_duplicates(subset=["machine_id", "ts"])
print(f"CSV {len(files)}개 | 중복 제거 후 {len(raw):,}행")
print(f"기간 {raw['ts'].min()} ~ {raw['ts'].max()}")

rule("A8-2. 결측률 (원본 수신 기준)")
print((raw[NUM].isna().mean() * 100).round(2).to_string())

rule("A8-3. ★ 수집 공백 — 통신이 끊긴 구간")
g = raw.sort_values(["machine_id", "ts"]).copy()
g["gap_min"] = g.groupby("machine_id")["ts"].diff().dt.total_seconds() / 60
gaps = g[g["gap_min"] > 2]
print(f"2분 넘는 공백 : {len(gaps):,}회")
print(f"최장 공백     : {gaps['gap_min'].max():.0f}분")
print(f"총 결손 시간  : {gaps['gap_min'].sum()/60:.1f}시간")
print("\n[설비별]")
print(gaps.groupby("machine_id")["gap_min"].agg(
    횟수="count", 최장="max", 합계시간=lambda s: round(s.sum()/60, 1)).to_string())

rule("A8-4. 단위 혼재 흔적")
n_c = int((raw["air_temp_k"] < 200).sum())
print(f"공기온도 200 K 미만 행: {n_c:,}건 ({n_c/len(raw)*100:.2f}%)")

rule("A8-5. ★ 결측률 vs 실제 손실률")
span = int((raw["ts"].max() - raw["ts"].min()).total_seconds() / 60) + 1
expected = span * raw["machine_id"].nunique()
print(f"있어야 할 행(분×설비) : {expected:,}")
print(f"실제 보유 행          : {len(raw):,}")
print(f"실제 손실률           : {(1-len(raw)/expected)*100:.2f}%")
print(f"단순 결측률(평균)     : {raw[NUM].isna().mean().mean()*100:.2f}%")
print("\n★ 두 숫자의 차이가 이 프로젝트에서 반복해 강조한 지점입니다.")
