"""
SQLite 적재 계층
================
설비 데이터는 "같은 시각 같은 설비"가 유일해야 합니다.
그래서 (machine_id, ts)에 UNIQUE 제약을 걸고 UPSERT로 넣습니다.
중복 전송이 와도 DB가 알아서 막아줍니다. 파이썬에서 막는 것보다 확실합니다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sensors.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id      TEXT    NOT NULL,
    ts              TEXT    NOT NULL,          -- ISO8601 문자열 (UTC 기준)
    type            TEXT,
    air_temp_k      REAL,
    process_temp_k  REAL,
    rot_speed_rpm   REAL,
    torque_nm       REAL,
    tool_wear_min   REAL,
    vibration_mms   REAL,
    current_a       REAL,
    humidity_pct    REAL,
    machine_failure INTEGER,
    collected_at    TEXT,
    UNIQUE (machine_id, ts)                    -- ★ 중복 방어선
);

CREATE INDEX IF NOT EXISTS ix_sensor_ts      ON sensor_raw (ts);
CREATE INDEX IF NOT EXISTS ix_sensor_machine ON sensor_raw (machine_id, ts);

CREATE TABLE IF NOT EXISTS collect_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT,
    window_start  TEXT,
    window_end    TEXT,
    rows_received INTEGER,
    rows_inserted INTEGER,
    rows_skipped  INTEGER,
    note          TEXT
);
"""

COLUMNS = [
    "machine_id",
    "ts",
    "type",
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mms",
    "current_a",
    "humidity_pct",
    "machine_failure",
    "collected_at",
]
# 컬럼 순서 고정 - DF의 컬럼 순서와 SQL문의 물음표 순서가 일치해야 함


def connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )  # DB 파일이 들어갈 폴더 없으면 미리 만들어놓음
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)  # SCHEMA 문자열 한 번에 실행
    return con


def upsert(con: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
    """(insert된 행 수, 중복으로 건너뛴 행 수)를 돌려줍니다."""
    df = df.reindex(columns=COLUMNS)  # DF의 컬럼을 정의한 COLUMNS 순서로 강제 재배열
    # 만약 원본 df에 없는 컬럼이 COLUMNS에 있다면, 그 컬럼은 자동으로 NaN으로 채워짐
    before = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    sql = (
        f"INSERT OR IGNORE INTO sensor_raw ({','.join(COLUMNS)}) "
        f"VALUES ({','.join('?' * len(COLUMNS))})"
    )  # SQL문 조립.
    con.executemany(
        sql, df.where(pd.notna(df), None).itertuples(index=False, name=None)
    )  # df.where(조건, 대체값)은 "조건이 True인 곳은 원래 값 유지, False인 곳은 대체값(None)으로 바꿔라"
    # pandas의 NaN을 파이썬의 None으로 바꾸는 작업
    # .itertuples(index=False, name=None): DataFrame을 한 행씩 튜플로 순회
    # con.executemany(sql, 튜플들): 같은 SQL문을 여러 데이터에 대해 반복 실행
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM sensor_raw").fetchone()[0]
    inserted = after - before  # 새로 들어간 행 수 계산
    return inserted, len(df) - inserted


def log_run(con, window_start, window_end, received, inserted, skipped, note=""):
    con.execute(
        "INSERT INTO collect_log (run_at, window_start, window_end,"
        " rows_received, rows_inserted, rows_skipped, note)"
        " VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
        (str(window_start), str(window_end), received, inserted, skipped, note),
    )
    con.commit()


def read_all(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM sensor_raw ORDER BY ts, machine_id", con)


# SQL 쿼리 결과를 바로 DataFrame으로 돌려줌 (전체 데이터를 시간 설비 순으로 정렬해서 가져옴)

# SCHEMA        → sensor_raw(중복방지+인덱스) + collect_log 테이블 정의
# COLUMNS       → INSERT 시 순서 고정용 리스트
# connect()     → 파일/폴더 준비 + 스키마 적용해서 연결 얻기
# upsert()      → NaN→None 변환, INSERT OR IGNORE로 안전하게 대량 삽입, 전후 카운트로 결과 집계
# log_run()     → collect_log에 실행 이력 한 줄 남기기
# read_all()    → 전체 데이터 조회
