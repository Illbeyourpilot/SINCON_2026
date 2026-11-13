"""D. 문학 주경기장 저녁 공연(PLAVE 9/12·13, 19시 시작) 보정.

기존 공연 보정은 문학역 '급증일'을 자동 감지해 평균하는데, 그 안에 낮 시범경기·포스트시즌이 섞여
저녁 공연의 22시 급증(실제 7천~9천 명)이 거의 반영되지 않는다 (v8d 22시 예측 86명·65명).
여기서는 22시 정점형 저녁 공연일(박효신 3회 + 2025-07-26)만으로 시간대별 편차를 재고,
PLAVE 날의 예측을 '평소 주말 예측 + w × 공연 편차'로 바꾼다.

사용: python post_concert.py <train.csv> <test.csv> <aux_data 폴더> <sincon_v2.py> <입력 제출.csv> <출력.csv> [w]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN, TEST, AUX, SRC, SUB, OUTF = sys.argv[1:7]
W = float(sys.argv[7]) if len(sys.argv) > 7 else 0.8
AUX = Path(AUX)

src = open(SRC, encoding="utf-8").read()
HSET = set(pd.to_datetime(re.findall(r'"(20\d\d-\d\d-\d\d)"', src.split("HOLIDAYS = pd.to_datetime([")[1].split("])")[0])))
tr = pd.read_csv(TRAIN, parse_dates=["date"])
te = pd.read_csv(TEST, parse_dates=["date"])
sub = pd.read_csv(SUB)
assert (sub.example_id.values == te.example_id.values).all()
geo = pd.read_csv(AUX / "station_geo_graph.csv").set_index("station_id")
ssg = set(pd.read_csv(AUX / "ssg_home_actual.csv", parse_dates=["date"]).date)
tr["lb"] = np.log1p(tr.boardings)
te["p"] = np.log1p(sub.boardings.to_numpy())

CONCERTS = pd.to_datetime(["2025-07-26", "2026-04-04", "2026-04-05", "2026-04-11"])   # 22시 정점 저녁 공연
TARGET = pd.to_datetime(["2026-09-12", "2026-09-13"])                                 # PLAVE, 19시 시작
# 문학 주변의 다른 급증일(경기·공연·축제)은 기준선에서 제외
OTHER_EV = set(pd.to_datetime(["2025-03-30", "2025-05-05", "2025-06-13", "2025-07-17", "2025-07-27", "2025-08-10",
                               "2025-09-06", "2025-09-14", "2025-09-27", "2025-10-09", "2025-10-10", "2025-10-11",
                               "2025-10-24", "2025-10-31", "2025-11-01", "2025-11-23", "2026-03-22",
                               "2026-08-29", "2026-08-30", "2026-10-24", "2026-10-25"]))
near = geo.index[geo.dist_munhak < 3000].tolist()


def off(d):
    return d.isin(HSET) | (d.dt.dayofweek >= 5)


def baseline(df, col, day, excl):
    """같은 역·시간, 같은 요일유형(주말·휴일)의 ±21일 비행사일 중앙값."""
    w = df[df.station_id.isin(near) & ((df.date - day).abs().dt.days <= 21) & (off(df.date) == off(pd.Series([day])).iloc[0])
           & ~df.date.isin(excl)]
    return w.groupby(["station_id", "hour"])[col].median()


excl = ssg | OTHER_EV | set(CONCERTS) | set(TARGET)
devs = []
for c in CONCERTS:
    b = baseline(tr, "lb", c, excl)
    x = tr[(tr.date == c) & tr.station_id.isin(near)].set_index(["station_id", "hour"]).lb
    devs.append((x - b).rename(c))
D = pd.concat(devs, axis=1)
dev = D.mean(axis=1)
# 효과가 뚜렷한 역만 (공연일 평균 편차가 어느 시간이든 0.3 이상)
keep = dev.groupby(level=0).max()
keep = keep[keep > 0.3].index.tolist()
print("공연 영향 역:", keep)
print("문학경기장(3127) 시간대별 평균 로그 편차:", dev.loc[3127].round(2).to_dict())

P = te.p.to_numpy().copy()
for day in TARGET:
    b = baseline(te, "p", day, excl)
    m = (te.date == day) & te.station_id.isin(keep)
    idx = pd.MultiIndex.from_frame(te.loc[m, ["station_id", "hour"]])
    newv = b.reindex(idx).to_numpy() + W * dev.reindex(idx).fillna(0.0).to_numpy()
    ok = ~np.isnan(newv)
    P[np.where(m)[0][ok]] = newv[ok]

s = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(np.clip(P, 0, None))})
assert len(s) == len(te) and s.boardings.notna().all()
s.to_csv(OUTF, index=False)
chg = np.abs(P - te.p.to_numpy()) > 1e-9
print(f"w={W} | 바뀐 행 {chg.sum()} -> {OUTF}")
v = te.assign(new=np.expm1(P), old=np.expm1(te.p))
print(v[(v.station_id == 3127) & v.date.isin(TARGET) & v.hour.between(18, 23)]
      .pivot(index="hour", columns="date", values=["old", "new"]).round(0).astype(int).to_string())
