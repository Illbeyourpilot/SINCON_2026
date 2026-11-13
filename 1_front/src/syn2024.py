"""2024년 시간대별 승차 합성: 일별 역 합계(실측) x 시간대 비율(모델).
비율 모델(LightGBM): 그 달 역x시간 월합계 비율(인천교통공사 월별 시간대 통계) + 요일유형/명절/날씨/SSG 경기 시각·관중/그날 합계의 평소 대비 편차.
검증: 2025-01~2026-06 train 에서 월 단위로 하나씩 빼고 학습 -> 그 달 합성 -> 실제와 RMSLE. 이후 2024-07~12 합성."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.argv = [sys.argv[0]] + sys.argv[1:]
MODE = sys.argv[1] if len(sys.argv) > 1 else "cv"
sys.path.insert(0, str(Path(__file__).resolve().parent))
_a = sys.argv; sys.argv = ["x"]
import sincon_v2 as V, hist_season as HS
sys.argv = _a
meta = pd.read_csv("data/metaData.csv")
tr = pd.read_csv("data/train.csv", parse_dates=["date"])[["date", "station_id", "hour", "boardings"]]
H24 = pd.to_datetime(["2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-01", "2024-05-05",
                      "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03",
                      "2024-10-09", "2024-12-25"])
HALL = set(V.HSET) | set(H24)
BIG = list(pd.to_datetime(["2024-02-10", "2024-09-17"])) + list(V.BIG_HOLIDAYS)


def blocks():
    off = lambda x: x in HALL or x.dayofweek >= 5
    out = {}
    for b in BIG:
        s_, e_ = b, b
        while off(s_ - pd.Timedelta(days=1)): s_ -= pd.Timedelta(days=1)
        while off(e_ + pd.Timedelta(days=1)): e_ += pd.Timedelta(days=1)
        out[b] = (s_, e_)
    return out


BL = blocks()


def holrel(d):
    rel = np.full(len(d), 99)
    for b, (s_, e_) in BL.items():
        r = (d - b).dt.days.to_numpy()
        ins = ((d >= s_) & (d <= e_)).to_numpy(); pre = ((d < s_) & (d >= s_ - pd.Timedelta(days=3))).to_numpy(); post = ((d > e_) & (d <= e_ + pd.Timedelta(days=3))).to_numpy()
        rel[ins] = np.clip(r[ins], -5, 5); rel[pre] = -20 - (s_ - d[pre]).dt.days.to_numpy(); rel[post] = 20 + (d[post] - e_).dt.days.to_numpy()
    return rel


# 월별 시간대 비율: 2024 이전은 인천교통공사 월별 통계, train 기간은 train 자체 월합계(같은 정의)
hm = HS.hist_table(meta)[["year", "month", "station_id", "hour", "count"]]
tm = tr.assign(year=tr.date.dt.year, month=tr.date.dt.month).groupby(["year", "month", "station_id", "hour"], as_index=False).boardings.sum().rename(columns={"boardings": "count"})
MS = pd.concat([hm[hm.year <= 2024], tm])
MS["msh"] = np.log(MS["count"] + 1) - np.log(MS.groupby(["year", "month", "station_id"])["count"].transform("sum") + 18)
# 날씨
def wx_load():
    a = json.load(open("ext/syn2024/weather_2024.json"))["hourly"]
    w = pd.DataFrame(a); w["t"] = pd.to_datetime(w.time)
    b = pd.read_csv("geo/weather_incheon_hourly.csv", parse_dates=["date"])
    w = w.assign(date=w.t.dt.normalize(), hour=w.t.dt.hour)[["date", "hour", "precipitation", "temperature_2m"]]
    b = b[["date", "hour", "precipitation", "temperature_2m"]] if "temperature_2m" in b else b.rename(columns={})[["date", "hour", "precipitation"]]
    return pd.concat([w[w.date < b.date.min()], b])
WX = wx_load()
box = pd.concat([pd.read_csv("ext/syn2024/ssg_box_2024.csv"), pd.read_csv("geo/ssg_box.csv")])
box = box[box.stadium.astype(str).str.contains("문학")].copy(); box["date"] = pd.to_datetime(box.date.astype(str), format="mixed")
box["endh"] = box.end.astype(str).str[:2].astype(float); box["sth"] = box.start.astype(str).str[:2].astype(float)
box = box.groupby("date").agg(endh=("endh", "max"), sth=("sth", "min"), crowd=("crowd", "sum"))
G = pd.read_csv("geo/station_geo_graph.csv").set_index("station_id") if Path("geo/station_geo_graph.csv").exists() else None


def feats(df, daily):
    """df: date, station_id, hour (+ boardings) ; daily: date, station_id, dayb"""
    z = df.merge(daily, on=["date", "station_id"], how="left")
    z["year"], z["month"] = z.date.dt.year, z.date.dt.month
    z = z.merge(MS[["year", "month", "station_id", "hour", "msh"]], on=["year", "month", "station_id", "hour"], how="left")
    dow = z.date.dt.dayofweek
    z["dt"] = np.where(z.date.isin(HALL) | (dow == 6), 2, np.where(dow == 5, 1, 0)); z["dow"] = dow
    z["hrel"] = holrel(z.date)
    z = z.merge(WX, on=["date", "hour"], how="left")
    # 그날 합계의 평소(같은 달·같은 요일유형 중앙값) 대비 편차
    dd = daily.copy(); dd["ym"] = dd.date.dt.to_period("M"); dw = dd.date.dt.dayofweek
    dd["dt"] = np.where(dd.date.isin(HALL) | (dw == 6), 2, np.where(dw == 5, 1, 0))
    dd["ddev"] = np.log(dd.dayb + 1) - np.log(dd.groupby(["station_id", "ym", "dt"]).dayb.transform("median") + 1)
    z = z.merge(dd[["date", "station_id", "ddev"]], on=["date", "station_id"], how="left")
    z["wkfrac"] = z.date.map(lambda d: 0)  # placeholder
    b = z.date.map(box.endh); z["ssg_k"] = np.where(b.notna(), z.hour - b, 99); z["ssg_crowd"] = z.date.map(box.crowd).fillna(0)
    z["ssg_st"] = (z.station_id == 3127).astype(int)
    z["lday"] = np.log(z.dayb + 1)
    return z


FC = ["station_id", "hour", "msh", "dt", "dow", "hrel", "precipitation", "ddev", "ssg_k", "ssg_crowd", "ssg_st", "lday", "month"]
P = dict(objective="l2", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)


def fit(z, rounds=600):
    y = np.log(z.boardings + 1) - np.log(z.dayb + 1)
    return lgb.train(P, lgb.Dataset(z[FC], y, categorical_feature=["station_id"]), rounds)


def synth(m, z):
    s = np.exp(m.predict(z[FC]))
    s = s / pd.Series(s).groupby([z.date.values, z.station_id.values]).transform("sum").to_numpy()
    return z.dayb.to_numpy() * s


daily_tr = tr.groupby(["date", "station_id"], as_index=False).boardings.sum().rename(columns={"boardings": "dayb"})
if MODE == "cv":
    Z = feats(tr, daily_tr)
    Z = Z[Z.msh.notna()]
    ym = Z.date.dt.to_period("M")
    rm = lambda a, b: np.sqrt(np.mean((np.log1p(a) - np.log1p(b)) ** 2))
    res = []
    for m_ in ["2025-02", "2025-05", "2025-08", "2025-10", "2026-01", "2026-04"]:
        te_ = ym == pd.Period(m_); m = fit(Z[~te_])
        zt = Z[te_]; sy = synth(m, zt)
        base = zt.dayb.to_numpy() * np.exp(zt.msh) / pd.Series(np.exp(zt.msh)).groupby([zt.date.values, zt.station_id.values]).transform("sum").to_numpy()
        res.append((m_, rm(sy, zt.boardings), rm(base, zt.boardings)))
        print(f"{m_}: 합성 RMSLE {res[-1][1]:.4f}  (단순 월비율 x 일합계 {res[-1][2]:.4f})", flush=True)
    print("평균", np.mean([r[1] for r in res]).round(4), np.mean([r[2] for r in res]).round(4))
else:
    # make <기준일> <태그>: 비율 모델은 기준일까지의 train 으로만 학습 (검증 폴드 누수 방지). A=2025-06-30, B=2026-02-28, full=2026-06-30
    cut, tag = pd.Timestamp(sys.argv[2]), sys.argv[3]
    d24 = pd.read_csv("ext/daily_hist/daily_board_2024h2.csv", parse_dates=["date"])
    d24 = d24[d24.date <= "2024-12-31"]
    Z = feats(tr, daily_tr); Z = Z[Z.msh.notna() & (Z.date <= cut)]
    m = fit(Z, 800)
    grid = d24[["date", "station_id"]].merge(pd.DataFrame({"hour": range(6, 24)}), how="cross")
    zt = feats(grid, d24); zt = zt[zt.msh.notna()]
    zt["boardings"] = synth(m, zt)
    # 하차 = 합성 승차 x 그 달 역x시간 하차/승차 비율 (인천교통공사 2024 월별 시간대 통계)
    d = pd.read_csv("data/ictr_hourly_monthly.csv"); d = d[d.year == 2024]
    d["n"] = d.station.map(HS.norm).replace(HS.FIX)
    mt = meta.assign(n=meta.station_name.map(HS.norm)); dupn = set(mt.n[mt.n.duplicated(keep=False)])
    a = d[~d.n.isin(dupn)].merge(mt[["n", "station_id"]], on="n")
    b = d[d.n.isin(dupn) & (d.line > 0)].merge(mt[["n", "canonical_line", "station_id"]], left_on=["n", "line"], right_on=["n", "canonical_line"])
    r = pd.concat([a, b]).pivot_table(index=["month", "station_id", "hour"], columns="kind", values="count", aggfunc="sum").reset_index()
    r["ratio"] = (r["하차"] + 1) / (r["승차"] + 1)
    zt = zt.merge(r[["month", "station_id", "hour", "ratio"]], on=["month", "station_id", "hour"], how="left")
    zt["alightings"] = (zt.boardings * zt.ratio.fillna(1.0)).round(1); zt["boardings"] = zt.boardings.round(1)
    zt[["date", "station_id", "hour", "boardings", "alightings"]].to_csv(f"ext/syn2024/syn_2024h2_{tag}.csv", index=False)
    print("합성 저장", tag, "비율모델 학습 ~", cut.date(), len(zt), zt.station_id.nunique(), "역")
