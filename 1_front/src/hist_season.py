"""과거(2016~2024) 월별 역 시간대별 통행량으로 계절 보정.
dh[s,h,m] = 연도 가중 평균 { log(월 m 하루평균) - log(같은 해 3~6월 하루평균) }   (코로나 2020·2021 제외, 최신 연도 가중)
dm[s,h,m] = log(모델 예측 월 m 하루평균) - log(기준일 직전 3~6월 실제 하루평균)
보정 c = w·(dh - dm) 를 그 역x시간x월의 모든 예측에 더함. 과거 자료는 2024-12 이전만 사용 (출처: parse_ictr.py)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIX = {"아시아드": "아시아드경기장", "가정시장": "가정중앙시장"}


def norm(s):
    return re.sub(r"\(.*?\)|역$|\s", "", str(s))


def hist_table(meta):
    cands = [Path(__file__).resolve().parent / "ictr_hourly_monthly.csv", ROOT / "ext" / "ictr_hourly_monthly.csv",
             Path.cwd() / "ictr_hourly_monthly.csv"] + sorted(Path("/kaggle/input").glob("**/ictr_hourly_monthly.csv"))
    d = pd.read_csv(next(c for c in cands if c.exists()))
    d = d[(d.kind == "승차") & d.hour.between(6, 23)].copy()
    d["n"] = d.station.map(norm).replace(FIX)
    m = meta.assign(n=meta.station_name.map(norm))
    dupn = set(m.n[m.n.duplicated(keep=False)])
    a = d[~d.n.isin(dupn)].merge(m[["n", "station_id"]], on="n")
    b = d[d.n.isin(dupn) & (d.line > 0)].merge(m[["n", "canonical_line", "station_id"]],
                                                left_on=["n", "line"], right_on=["n", "canonical_line"])
    h = pd.concat([a, b])[["year", "month", "station_id", "hour", "count"]]
    h = h.groupby(["year", "month", "station_id", "hour"], as_index=False)["count"].sum()
    h["days"] = pd.to_datetime(dict(year=h.year, month=h.month, day=1)).dt.days_in_month
    h["lv"] = np.log(h["count"] / h.days + 1.0)
    return h


def hist_delta(h, years, decay):
    """연도 가중 (최신 연도 가중치 1, 1년 이전마다 x decay) 계절 변화."""
    h = h[h.year.isin(years)]
    ref = h[h.month.between(3, 6)].groupby(["year", "station_id", "hour"]).lv.mean().rename("ref")
    x = h[h.month.between(7, 10)].join(ref, on=["year", "station_id", "hour"])
    x["dh"] = x.lv - x.ref
    x["w"] = decay ** (max(years) - x.year)
    x = x.dropna(subset=["dh"])
    g = x.assign(wd=x.w * x.dh).groupby(["station_id", "hour", "month"])[["wd", "w"]].sum()
    return (g.wd / g.w).rename("dh").reset_index()


def correction(pred, actual, anchor, dh, w, pool="sh", clip=0.3):
    """pred: [date, station_id, hour, P(log1p)], actual: train (lb 포함). 반환: 행별 보정값."""
    a = pd.Timestamp(anchor)
    ref = actual[(actual.date > a - pd.DateOffset(months=4)) & (actual.date <= a)]
    ref = ref.groupby(["station_id", "hour"]).boardings.mean().rename("r")
    p = pred.assign(month=pred.date.dt.month, e=np.expm1(pred.P))
    pm = p.groupby(["station_id", "hour", "month"]).e.mean().rename("pm").reset_index()
    pm = pm.join(ref, on=["station_id", "hour"]).merge(dh, on=["station_id", "hour", "month"], how="left")
    pm["dm"] = np.log(pm.pm + 1.0) - np.log(pm.r + 1.0)
    pm["gap"] = pm.dh - pm.dm
    if pool == "s":   # 역x월 단위로 평균 (시간대 잡음 줄임)
        pm["gap"] = pm.groupby(["station_id", "month"]).gap.transform("mean")
    elif pool == "lh":  # 시간x월 전체 역 평균
        pm["gap"] = pm.groupby(["hour", "month"]).gap.transform("mean")
    pm["c"] = (w * pm.gap).clip(-clip, clip).fillna(0.0)
    return p[["station_id", "hour", "month"]].merge(pm[["station_id", "hour", "month", "c"]], how="left").c.fillna(0.0).to_numpy()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import struct_diag as SD
    V = SD.HC.V
    meta = pd.read_csv(ROOT / "data" / "metaData.csv")
    tr = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    h = hist_table(meta)
    print("과거 자료 역 수", h.station_id.nunique(), "연도", sorted(h.year.unique()))
    for F in ("A",):
        m = SD.blend(F)
        y, P = m.lb.to_numpy(), m.P.to_numpy()
        r = lambda q: np.sqrt(np.mean((q - y) ** 2))  # noqa: E731
        a = V.FOLDS[F][0]
        print(f"[{F}] 현재 {r(P):.5f}")
        for yrs, nm in (([2016, 2017, 2018, 2019, 2022, 2023, 2024], "16~19+22~24"), ([2022, 2023, 2024], "22~24"),
                        ([2017, 2018, 2019, 2022, 2023, 2024], "17~19+22~24")):
            for decay in (1.0, 0.7):
                dh = hist_delta(h, yrs, decay)
                for pool in ("sh", "s", "lh"):
                    res = []
                    for w in (0.25, 0.5, 0.75, 1.0):
                        c = correction(m.assign(P=P), tr, a, dh, w, pool)
                        res.append(f"w{w}:{r(P + c) - r(P):+.5f}")
                    print(f"  {nm:12s} 감쇠{decay} 단위{pool:2s} " + " ".join(res), flush=True)
