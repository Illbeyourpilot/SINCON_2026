"""일 총량 x 시간대 비중 분해 모델.
1) 역별 일 승차 총량(log)을 다중 기준일 LightGBM 으로 예측: y = log일총량(t) - 기준일 직전 28일 같은 요일유형 평균(log)
2) 시간대 비중 = 기준일 직전 56일 같은 요일유형(역별) 평균 비중 (공휴일은 일요일 비중)
사용: python daily_model.py cv            -> out/cv_daily_{A,B,C}.csv
      python daily_model.py full          -> testpred_daily_full.csv
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "."); import sincon_v2 as V

MODE = sys.argv[1]
tr = pd.read_csv("train.csv", parse_dates=["date"])
te = pd.read_csv("test.csv", parse_dates=["date"])
geo = pd.read_csv("station_geo_graph.csv", encoding="utf-8-sig").set_index("station_id")
HS = set(V.HSET); BIG = set(V.BIG_HOLIDAYS)
BIGW = set()
for a, b in [("2025-01-27", "2025-01-30"), ("2025-10-03", "2025-10-09"), ("2026-02-14", "2026-02-18"), ("2026-09-24", "2026-09-27")]:
    BIGW |= set(pd.date_range(a, b))
NEW = [3107, 3108, 3109]; OPEN = pd.Timestamp("2025-06-28")

w = pd.read_csv("weather_incheon_hourly.csv", parse_dates=["date"]); w = w[w.hour.between(6, 23)]
wd_ = w.groupby("date").agg(tmean=("temperature_2m", "mean"), tmax=("temperature_2m", "max"), rain=("precipitation", "sum"),
                            rain_h=("precipitation", lambda s: (s > 0.5).sum()), rain_ev=("precipitation", lambda s: s.iloc[11:].sum()))
ssg = pd.read_csv("ssg_home_actual.csv", parse_dates=["date"]).set_index("date").start_hour
bfc = pd.read_csv("bucheon_home_actual.csv", parse_dates=["date"]).set_index("date").start_hour
nearM = set(geo.index[geo.dist_munhak < 2500]); nearB = set(geo.index[geo.dist_bucheon_stadium < 2500])


def inrng(d, rngs):
    m = np.zeros(len(d), bool)
    for a, b in rngs: m |= ((d >= a) & (d <= b)).to_numpy()
    return m


def dtype(d):  # 0 평일 1 토 2 일/공휴일
    return np.where(d.isin(HS) | (d.dt.dayofweek == 6), 2, np.where(d.dt.dayofweek == 5, 1, 0))


D = tr.groupby(["station_id", "date"]).boardings.sum().rename("tot").reset_index()
D["ly"] = np.log1p(D.tot); D["t"] = dtype(D.date)
H = tr.copy(); H["t"] = dtype(H.date)
H = H.merge(D[["station_id", "date", "tot"]], on=["station_id", "date"]); H["sh"] = H.boardings / H.tot.clip(lower=1)
STATIONS = sorted(D.station_id.unique())


def level(a):  # 기준일 직전 28일 요일유형별 log 평균
    x = D[(D.date <= a) & (D.date > a - pd.Timedelta(days=28)) & ~D.date.isin(HS)]
    lv = x.groupby(["station_id", "t"]).ly.mean().unstack()
    lv[2] = lv[2].fillna(lv[1]); return lv


def frame(a, dates):
    lv = level(a)
    f = pd.MultiIndex.from_product([STATIONS, dates], names=["station_id", "date"]).to_frame(index=False)
    f["t"] = dtype(f.date)
    f["base"] = [lv.at[s, t] if s in lv.index else np.nan for s, t in zip(f.station_id, f.t)]
    f["base_wd"] = f.station_id.map(lv[0])
    f["h"] = (f.date - a).dt.days
    f["dow"] = f.date.dt.dayofweek; f["doy"] = f.date.dt.dayofyear
    f["hol"] = f.date.isin(HS).astype(int); f["bigw"] = f.date.isin(BIGW).astype(int)
    f["hol_prev"] = (f.date + pd.Timedelta(days=1)).isin(HS).astype(int); f["hol_next"] = (f.date - pd.Timedelta(days=1)).isin(HS).astype(int)
    f["univ"] = inrng(f.date, V.UNIV_VAC).astype(int); f["school"] = inrng(f.date, V.SCHOOL_VAC).astype(int)
    f = f.join(wd_, on="date")
    f["ssg"] = (f.date.isin(ssg.index) & f.station_id.isin(nearM)).astype(int)
    f["bfc"] = (f.date.isin(bfc.index) & f.station_id.isin(nearB)).astype(int)
    f["new"] = f.station_id.isin(NEW).astype(int); f["since_open"] = np.where(f.new == 1, (f.date - OPEN).dt.days, -1)
    for c in ["dist_munhak", "dist_songdo_park", "gdist_seoul", "is_transfer"]: f[c] = f.station_id.map(geo[c])
    return f


FE = ["station_id", "t", "h", "dow", "hol", "bigw", "hol_prev", "hol_next", "univ", "school", "tmean", "tmax", "rain", "rain_h",
      "rain_ev", "ssg", "bfc", "new", "since_open", "dist_munhak", "dist_songdo_park", "gdist_seoul", "is_transfer", "base_wd"]
PRM = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0)


def train_set(cut):
    rows = []
    for a in pd.date_range("2025-02-01", cut - pd.Timedelta(days=7), freq="7D"):
        dates = pd.date_range(a + pd.Timedelta(days=1), min(a + pd.Timedelta(days=123), cut))
        f = frame(a, dates).merge(D[["station_id", "date", "ly"]], on=["station_id", "date"])
        rows.append(f)
    x = pd.concat(rows); x = x[x.base.notna()]; x["y"] = x.ly - x.base; return x


def profile(a, dates, stations):
    x = H[(H.date <= a) & (H.date > a - pd.Timedelta(days=56)) & ~H.date.isin(HS)]
    pr = x.groupby(["station_id", "t", "hour"]).sh.mean()
    q = pd.MultiIndex.from_product([stations, dates, range(6, 24)], names=["station_id", "date", "hour"]).to_frame(index=False)
    q["t"] = dtype(q.date)
    q["sh"] = pr.reindex(pd.MultiIndex.from_frame(q[["station_id", "t", "hour"]])).to_numpy()
    return q


def run(a, dates, cut):
    x = train_set(cut)
    m = lgb.train(PRM, lgb.Dataset(x[FE], x.y, categorical_feature=["station_id"]), 800)
    f = frame(a, dates); f["pd"] = f.base + m.predict(f[FE])
    q = profile(a, dates, STATIONS).merge(f[["station_id", "date", "pd"]], on=["station_id", "date"])
    q["pred"] = np.log1p(np.expm1(q.pd) * q.sh)
    return q[["date", "station_id", "hour", "pred"]]


if MODE == "cv":
    for k, (a, e) in V.FOLDS.items():
        a, e = pd.Timestamp(a), pd.Timestamp(e)
        out = run(a, pd.date_range(a + pd.Timedelta(days=1), e), a)
        out.to_csv(f"out/cv_daily_{k}.csv", index=False); print(k, len(out), flush=True)
else:
    a = tr.date.max()
    out = run(a, pd.date_range(te.date.min(), te.date.max()), a)
    te.merge(out, on=["date", "station_id", "hour"], how="left")[["example_id", "date", "station_id", "hour", "pred"]] \
        .rename(columns={"pred": "pred_log"}).to_csv("testpred_daily_full.csv", index=False)
    print("full done")
