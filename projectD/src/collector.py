"""
주기 수집기
===========
GitHub Actions가 하루 한 번 이 파일을 실행합니다.

    python src/collector.py --minutes 1440

동작
  1) 센서 소스에서 최근 N분 구간을 받아온다
  2) data/history/YYYY-MM-DD.csv 로 원본 그대로 저장 (감사 추적)
  3) SQLite에 UPSERT (중복은 DB가 막음)
  4) collect_log에 수집 이력을 남긴다

★ 왜 CSV와 SQLite를 둘 다 쓰나
   - CSV: git에 커밋해도 diff가 보인다. "언제 뭐가 들어왔는지" 추적 가능
   - SQLite: 조회·조인이 편하다. 하지만 바이너리라 git에 넣으면 diff가 안 보인다
   그래서 CSV만 커밋하고, DB는 CSV로부터 언제든 재생성합니다(build_db.py).
   이 구조를 면접에서 설명하면 "재현 가능한 파이프라인"을 아는 사람으로 보입니다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import db as dbmod              # noqa: E402
from simulator import sample_window   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"


def fetch(minutes: int, end: str | None = None) -> pd.DataFrame:
    """센서 소스 호출부. 실제 현장이라면 여기가 REST API 호출이 됩니다.

    예)  r = requests.get(API, params={...}, timeout=10)
         r.raise_for_status()
         return pd.DataFrame(r.json()["items"])
    """
    return sample_window(n_minutes=minutes, end=end)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=1440, help="수집할 구간 길이(분)")
    ap.add_argument("--end", default=None, help="구간 끝 시각(기본: 지금)")
    ap.add_argument("--db", default=str(dbmod.DB_PATH))
    ap.add_argument("--hist", default=str(HIST), help="일별 CSV 저장 폴더")
    args = ap.parse_args()

    hist = Path(args.hist)
    hist.mkdir(parents=True, exist_ok=True)

    try:
        raw = fetch(args.minutes, args.end)
    except Exception as e:                       # 수집 실패해도 워크플로는 죽지 않게
        print(f"[ERROR] 수집 실패: {type(e).__name__}: {e}")
        return 1

    if raw.empty:
        print("[WARN] 받은 데이터가 0건입니다. 종료합니다.")
        return 0

    w_start, w_end = raw["ts"].min(), raw["ts"].max()
    tag = pd.Timestamp(w_end).strftime("%Y-%m-%d")
    csv_path = hist / f"{tag}.csv"

    # 같은 날 여러 번 돌아도 안전하게: 기존 파일과 합쳐 중복 제거
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        raw = pd.concat([old, raw], ignore_index=True)
    raw = raw.drop_duplicates(subset=["machine_id", "ts"], keep="last")
    raw.to_csv(csv_path, index=False)

    con = dbmod.connect(args.db)
    inserted, skipped = dbmod.upsert(con, raw)
    dbmod.log_run(con, w_start, w_end, len(raw), inserted, skipped,
                  note=f"csv={csv_path.name}")
    total = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    con.close()

    print(f"[OK] window {w_start} ~ {w_end}")
    print(f"     받은 행 {len(raw):,} / DB 신규 {inserted:,} / 중복 스킵 {skipped:,}")
    print(f"     CSV  {csv_path}")
    print(f"     DB 누적 {total:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
