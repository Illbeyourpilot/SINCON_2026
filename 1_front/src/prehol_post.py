"""명절 연휴 직전 평일(귀성 출발일) 보정: 2026-09-23.
과거 3건(2025-01-24, 2025-10-02, 2026-02-13)의 시간대별 (당일 / 앞 3주 같은 요일 평균) 비율을
우리 예측의 같은 비율과 맞추도록 log 보정 x PH_K(기본 0.7). 신규역 제외. 사용: python prehol_post.py in.csv out.csv"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
K = float(os.environ.get("PH_K", 0.7))
NEW = [3107, 3108, 3109]
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"]); te = pd.read_csv(ROOT / "data/test.csv", parse_dates=["date"])
b = pd.read_csv(sys.argv[1]); assert (b.example_id == te.example_id).all(); te["b"] = b.boardings.to_numpy()


def hr(df, col, day, ref):
    df = df[~df.station_id.isin(NEW)]
    a = df[df.date == day].groupby("hour")[col].sum()
    r = df[df.date.isin(pd.to_datetime(ref))].groupby(["date", "hour"])[col].sum().groupby("hour").mean()
    return a / r


EV = {"2025-01-24": ["2025-01-03", "2025-01-10", "2025-01-17"], "2025-10-02": ["2025-09-11", "2025-09-18", "2025-09-25"],
      "2026-02-13": ["2026-01-23", "2026-01-30", "2026-02-06"]}
hist = pd.concat([hr(tr, "boardings", pd.Timestamp(d), r) for d, r in EV.items()], axis=1).mean(axis=1)
day, ref = pd.Timestamp("2026-09-23"), ["2026-09-02", "2026-09-09", "2026-09-16"]
adj = K * np.log(hist / hr(te, "b", day, ref))
m = (te.date == day) & ~te.station_id.isin(NEW)
x = np.log1p(te.b) + np.where(m, te.hour.map(adj).fillna(0), 0)
print("시간대별 보정(log):", adj.round(3).to_dict())
b.assign(boardings=np.expm1(np.clip(x, 0, None))).to_csv(sys.argv[2], index=False); print("저장", sys.argv[2])
