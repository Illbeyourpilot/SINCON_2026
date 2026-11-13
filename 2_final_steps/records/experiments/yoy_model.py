"""작년 대비 변화량 모델(YoY delta LGB).
analog lm(t) = 작년 같은 요일(t-357/364/371, 공휴일 lag 제외) log 승차 중앙값, g_s = 역별 성장률(평일 수준 YoY, 전체와 0.5 축소)
target  y = lb(t) - lm(t) - g_s ; 피처 = 올해/작년 날씨·공휴일·경기·방학 차이 + 역·시간·요일
pred = lm + g_s + f(x)
사용: python yoy_model.py cv   -> out/cv_yoy_B.csv  (학습 2026-01~02, 검증 03~06)
      python yoy_model.py full -> testpred_yoy_full.csv (학습 2026-01~06)
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "."); import sincon_v2 as V

MODE = sys.argv[1]
tr = pd.read_csv("train.csv", parse_dates=["date"]); tr["lb"] = np.log1p(tr.boardings)
te = pd.read_csv("test.csv", parse_dates=["date"])
geo = pd.read_csv("station_geo_graph.csv", encoding="utf-8-sig").set_index("station_id")
HS = set(V.HSET); NEW = [3107, 3108, 3109]
BIGW = set()
for a, b in [("2025-01-25", "2025-02-02"), ("2025-10-02", "2025-10-10"), ("2026-02-13", "2026-02-19"), ("2026-09-23", "2026-09-28")]:
    BIGW |= set(pd.date_range(a, b))
W = pd.read_csv("weather_incheon_hourly.csv", parse_dates=["date"]).set_index(["date", "hour"])
Wd = W.reset_index(); Wd = Wd[Wd.hour.between(6, 23)].groupby("date").agg(rain_d=("precipitation", "sum"), t_d=("temperature_2m", "mean"))
ssg = set(pd.read_csv("ssg_home_actual.csv", parse_dates=["date"]).date)
bfc = set(pd.read_csv("bucheon_home_actual.csv", parse_dates=["date"]).date)
nearM = set(geo.index[geo.dist_munhak < 3000]); nearB = set(geo.index[geo.dist_bucheon_stadium < 3000])
L = tr.set_index(["station_id", "date", "hour"]).lb
LAGS = (357, 364, 371)


def inrng(d, rngs):
    m = np.zeros(len(d), bool)
    for a, b in rngs: m |= ((d >= a) & (d <= b)).to_numpy()
    return m


def growth(lo26, hi26, lo25, hi25):
    h = tr[(tr.date.dt.dayofweek < 5) & ~tr.date.isin(HS)]
    def lv(lo, hi): x = h[h.date.between(lo, hi)]; return x.groupby("station_id").boardings.sum() / x.date.nunique()
    g = np.log(lv(lo26, hi26) / lv(lo25, hi25))
    gl = np.log(h[h.date.between(lo26, hi26)].boardings.mean() / h[h.date.between(lo25, hi25)].boardings.mean())
    return 0.5 * g + 0.5 * gl


def build(df, gs):
    f = df[["date", "station_id", "hour"]].copy()
    vals = []
    for lag in LAGS:
        ld = f.date - pd.Timedelta(days=lag)
        v = L.reindex(pd.MultiIndex.from_arrays([f.station_id, ld, f.hour])).to_numpy().copy()
        v[(ld.isin(HS) | ld.isin(BIGW)).to_numpy()] = np.nan; vals.append(v)
    V3 = np.vstack(vals)
    f["lm"] = np.nanmedian(V3, axis=0); f["lsd"] = np.nanstd(V3, axis=0)
    f["g"] = f.station_id.map(gs).to_numpy()
    ly = f.date - pd.Timedelta(days=364)
    f["dow"] = f.date.dt.dayofweek; f["we"] = (f.dow >= 5).astype(int)
    f["hol"] = f.date.isin(HS).astype(int); f["bigw"] = f.date.isin(BIGW).astype(int)
    f["hol_adj"] = ((f.date + pd.Timedelta(days=1)).isin(HS) | (f.date - pd.Timedelta(days=1)).isin(HS)).astype(int)
    f["ly_hol_adj"] = sum(((f.date - pd.Timedelta(days=l) + pd.Timedelta(days=s)).isin(HS)) for l in LAGS for s in (-1, 1)).astype(int)
    f["univ"] = inrng(f.date, V.UNIV_VAC).astype(int); f["univ_ly"] = inrng(ly, V.UNIV_VAC).astype(int)
    f["school"] = inrng(f.date, V.SCHOOL_VAC).astype(int); f["school_ly"] = inrng(ly, V.SCHOOL_VAC).astype(int)
    wi = pd.MultiIndex.from_arrays([f.date, f.hour]); wl = pd.MultiIndex.from_arrays([ly, f.hour])
    f["rain"] = W.precipitation.reindex(wi).to_numpy(); f["rain_ly"] = W.precipitation.reindex(wl).to_numpy()
    f["temp"] = W.temperature_2m.reindex(wi).to_numpy(); f["temp_ly"] = W.temperature_2m.reindex(wl).to_numpy()
    f["rain_d"] = f.date.map(Wd.rain_d).to_numpy(); f["rain_d_ly"] = ly.map(Wd.rain_d).to_numpy()
    f["drain"] = f.rain_d - f.rain_d_ly; f["dtemp"] = f.temp - f.temp_ly
    f["game"] = ((f.date.isin(ssg) & f.station_id.isin(nearM)) | (f.date.isin(bfc) & f.station_id.isin(nearB))).astype(int)
    f["game_ly"] = sum((((f.date - pd.Timedelta(days=l)).isin(ssg) & f.station_id.isin(nearM)) |
                        ((f.date - pd.Timedelta(days=l)).isin(bfc) & f.station_id.isin(nearB))) for l in LAGS).astype(int)
    for c in ["dist_munhak", "dist_songdo_park", "gdist_seoul", "is_transfer"]: f[c] = f.station_id.map(geo[c]).to_numpy()
    return f


FE = ["station_id", "hour", "dow", "we", "hol", "bigw", "hol_adj", "ly_hol_adj", "univ", "univ_ly", "school", "school_ly", "rain", "rain_ly",
      "temp", "temp_ly", "rain_d", "rain_d_ly", "drain", "dtemp", "game", "game_ly", "lsd", "lm", "dist_munhak", "dist_songdo_park",
      "gdist_seoul", "is_transfer"]
PRM = dict(objective="huber", alpha=0.5, learning_rate=0.03, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.7,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)


def fit(lo, hi, gs, rounds=600):
    x = tr[tr.date.between(lo, hi) & ~tr.station_id.isin(NEW)][["date", "station_id", "hour", "lb"]]
    f = build(x, gs); f["y"] = x.lb.to_numpy() - f.lm - f.g; f = f[f.y.notna()]
    return lgb.train(PRM, lgb.Dataset(f[FE], f.y, categorical_feature=["station_id"]), rounds)


if MODE == "cv":
    gs = growth("2026-01-01", "2026-02-28", "2025-01-01", "2025-02-28")
    m = fit("2026-01-01", "2026-02-28", gs)
    q = tr[tr.date.between("2026-03-01", "2026-06-30")][["date", "station_id", "hour"]]
    f = build(q, gs); q["pred"] = f.lm + f.g + m.predict(f[FE]); q["base"] = f.lm + f.g
    q.to_csv("out/cv_yoy_B.csv", index=False); print("cv done", q.pred.notna().sum())
else:
    gs = growth("2026-05-01", "2026-06-30", "2025-05-01", "2025-06-30")
    m = fit("2026-01-01", "2026-06-30", gs)
    f = build(te, gs); te["pred_log"] = f.lm + f.g + m.predict(f[FE]); te["base"] = f.lm + f.g
    te[["example_id", "date", "station_id", "hour", "pred_log", "base"]].to_csv("testpred_yoy_full.csv", index=False); print("full done")
