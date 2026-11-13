"""최종 후처리: 모델 예측(testpred_*.csv) 혼합 + 보정 단계 -> submission.csv.

모델: 트리 = 0.75·LightGBM(lt1·lt2 평균, 날씨 반영) + 0.25·XGBoost(v3 x10·x8 평균)
      Chronos = 0.5·Chronos-2(위치 공변량) + 0.5·Chronos-2(+날씨·실제 경기)
      P = 0.5·트리 + 0.5·Chronos
보정 (모두 기준일 6/30 이전 승차 자료로 학습, 7/1 이후는 승차가 아닌 메타데이터만 사용):
  1. 날씨(비·눈) 보정      : 평소 대비 편차 ~ 강수 (LightGBM, 단조 제약)                       x1.0
  2. 경기 보정            : SSG·부천FC 실제 경기일, 경기장 3km 이내 역, 시작 대비 시간별 평균 편차  x0.4
  3. 명절 보정            : 추석 ±3일은 과거 명절 패턴 목표와 반반
  4. 공연·축제 보정        : 공표된 행사일, 행사장 2km 이내 역의 과거 행사일 시간별 편차  x1.0~1.5 (검증 A·B)
  5. 신규역 보정          : 신규역 트리 예측의 수준을 기존역과 같은 계절 변화로 절반 맞춤
사용: python final_post.py <작업 폴더>   (testpred_*.csv 와 보조 csv 가 있는 곳)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
sys.argv = sys.argv[:1]
sys.path.insert(0, str(WORK))
import sincon_v2 as V  # noqa: E402

tr, te, meta, geo = V.load()
G = geo.set_index("station_id")
NEW = [3107, 3108, 3109]
K = ["date", "station_id", "hour"]
ANCHOR = tr.date.max()


def typ(d):
    return (d.date.isin(V.HSET) | (d.date.dt.dayofweek >= 5)).astype(int)


def rolling_dev(h, win):
    h = h.sort_values("date").copy()
    h["base"] = h.groupby(["station_id", "hour", "off"]).lb.transform(
        lambda s: s.rolling(win, center=True, min_periods=(5 if win == 11 else 8)).median())
    h["dev"] = h.lb - h.base
    return h


def load_pred(tag):
    d = te[["example_id"]].merge(pd.read_csv(WORK / f"testpred_{tag}.csv"), on="example_id", how="left")
    assert d.pred_log.notna().all(), tag
    return d.pred_log.to_numpy()


# ---------------- 1. 날씨 보정 ----------------
def wx_correction():
    W = pd.read_csv(V._find("weather_incheon_hourly.csv"), parse_dates=["date"]).sort_values(["date", "hour"])
    W["p3"] = W.precipitation.rolling(3, min_periods=1).sum()
    W = W.merge(W[W.hour.between(6, 23)].groupby("date").precipitation.sum().rename("pday"), on="date")
    W = W.rename(columns={"precipitation": "p1", "snowfall": "snow"})
    WX = ["p1", "p3", "pday", "snow"]
    FE = WX + ["hour", "off"]

    def prep(df):
        d = df.merge(W[["date", "hour"] + WX], on=["date", "hour"], how="left")
        d["off"] = typ(d)
        d["month"] = d.date.dt.month
        return d
    h = rolling_dev(prep(tr[["date", "station_id", "hour", "lb"]]), 11)
    h = h[h.base.notna()]
    h = h[h.dev.abs() < 2]
    g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=300,
                          monotone_constraints=[-1, -1, -1, -1, 0, 0], verbose=-1).fit(h[FE], h.dev)
    h["g"] = g.predict(h[FE])
    ref = h.groupby(["station_id", "hour", "off", "month"]).g.mean().rename("gref")
    ref2 = h.groupby(["station_id", "hour", "off"]).g.mean().rename("gref2")
    d = prep(te[K].copy())
    d["g"] = g.predict(d[FE])
    d = d.merge(ref, on=["station_id", "hour", "off", "month"], how="left").merge(
        ref2, on=["station_id", "hour", "off"], how="left")
    return (d.g - d.gref.fillna(d.gref2).fillna(0.0)).to_numpy()


# ---------------- 2. 경기 보정 ----------------
def game_correction():
    evs = [(pd.read_csv(V._find(f"{f}_home_actual.csv"), parse_dates=["date"]).set_index("date").start_hour, col)
           for f, col in (("ssg", "dist_munhak"), ("bucheon", "dist_bucheon_stadium"))]

    def tag(d):
        kind, rel = np.full(len(d), -1), np.full(len(d), 99)
        for i, (s, col) in enumerate(evs):
            st = d.date.map(s).to_numpy(float)
            r = d.hour.to_numpy() - st
            hit = ~np.isnan(st) & (d.station_id.map(G[col]).to_numpy() < 3000) & (r >= -3) & (r <= 6)
            kind[hit], rel[hit] = i, r[hit].astype(int)
        return d.assign(kind=kind, rel=rel, off=typ(d))
    h = rolling_dev(tag(tr[["date", "station_id", "hour", "lb"]].copy()), 11)
    h = h[(h.kind >= 0) & h.base.notna()]
    g = h.groupby(["station_id", "off", "kind", "rel"]).dev.agg(["sum", "count"])
    T = (g["sum"] / (g["count"] + 3.0)).rename("ev").reset_index()
    d = tag(te[K].copy()).merge(T, on=["station_id", "off", "kind", "rel"], how="left")
    return np.where(d.kind.to_numpy() >= 0, d.ev.fillna(0.0).to_numpy(), 0.0)


# ---------------- 3. 명절 보정 (목표값) ----------------
def holiday_target():
    def rel_of(dates):
        rel = pd.Series(99, index=dates.index)
        for b in V.BIG_HOLIDAYS:
            r = (dates - b).dt.days
            rel = rel.where(~((r.abs() <= 3) & (rel == 99)), r)
        return rel

    def lvl(h, end):
        x = h[(h.date > end - pd.Timedelta(days=28)) & (h.date <= end)]
        return x.groupby(["station_id", "hour", "t"]).lb.median().rename("lvl")
    h = tr[["date", "station_id", "hour", "lb"]].copy()
    h["t"] = typ(h)
    devs = []
    for b in V.BIG_HOLIDAYS:
        if b + pd.Timedelta(days=3) > ANCHOR:
            continue
        x = h[(h.date - b).dt.days.abs() <= 3].copy()
        x["rel"] = (x.date - b).dt.days
        x = x.merge(lvl(h, b - pd.Timedelta(days=11)).reset_index(), on=["station_id", "hour", "t"], how="left")
        x["dev"] = x.lb - x.lvl
        devs.append(x[["station_id", "hour", "rel", "dev"]])
    D = pd.concat(devs).groupby(["station_id", "hour", "rel"]).dev.mean().rename("hdev").reset_index()
    d = te[K].copy()
    d["t"], d["rel"] = typ(d), rel_of(d.date)
    d = d.merge(D, on=["station_id", "hour", "rel"], how="left").merge(
        lvl(h, ANCHOR).reset_index(), on=["station_id", "hour", "t"], how="left")
    tg = (d.lvl + d.hdev).to_numpy()
    return np.where((d.rel != 99).to_numpy() & ~np.isnan(tg), tg, np.nan)


# ---------------- 4. 공연·축제 보정 ----------------
VENUES = {"munhak": [3127], "asiad": [3209, 3210], "songdo": [3139], "bupyeong": [3119]}
# 과거 행사일을 급증 감지 대신 공표 일정으로 지정하는 행사장 (부평풍물대축제 2025: 9/26~28, 부평대로 일대)
HIST_DAYS = {"bupyeong": ["2025-09-26", "2025-09-27", "2025-09-28"]}
EVENTS = {  # 공표 일정 (웹): (행사장, 날짜들, 배율)  배율: 공연 1.0, 낮 페스티벌 1.0, 송도 축제 0.5 (공개 점수로 확인)
    # 부평풍물대축제(10/2~4)는 공개 점수에서 효과 없어 제외
    "munhak": (["2026-09-12", "2026-09-13"], 1.0),                                        # PLAVE 콘서트
    "munhak_fest": (["2026-08-29", "2026-08-30", "2026-10-24", "2026-10-25"], 1.0),        # Sustainable Wave, GHOSTIVAL
    "asiad": (["2026-07-11"], 1.0),                                                        # 싸이 흠뻑쇼
    "songdo": (["2026-07-31", "2026-08-01", "2026-08-02", "2026-08-08", "2026-08-15",      # 펜타포트, 송도해변축제
                "2026-08-22", "2026-08-23", "2026-08-24", "2026-08-25", "2026-08-26",      # 송도맥주축제
                "2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30", "2026-09-12"], 0.5),  # 이슬라이브
}


def venue_correction():
    ssg = set(pd.read_csv(V._find("ssg_home_actual.csv"), parse_dates=["date"]).date)
    h = tr[["date", "station_id", "hour", "lb", "boardings"]].copy()
    h["off"] = typ(h)
    h = rolling_dev(h, 21)
    T = {}
    for v, reps in VENUES.items():
        if v in HIST_DAYS:
            days = list(pd.to_datetime(HIST_DAYS[v]))
        else:
            r = h[h.station_id.isin(reps) & (h.dev > 1.5) & (h.boardings > 300)]
            if v == "munhak":
                r = r[~r.date.isin(ssg)]
            days = sorted(set(r.date))
        lat, lon = G.loc[reps[0], "lat"], G.loc[reps[0], "lon"]
        near = [s for s in G.index if V.hav(G.loc[s, "lat"], G.loc[s, "lon"], lat, lon) < 2000]
        x = h[h.date.isin(days) & h.station_id.isin(near)]
        g = x.groupby(["station_id", "hour"]).dev.agg(["sum", "count"])
        T[v] = (g["sum"] / (g["count"] + 2.0)).rename("vn").reset_index()
        print(f"  행사장 {v}: 과거 행사일 {len(days)}일, 주변역 {len(near)}개", flush=True)
    out = np.zeros(len(te))
    for k, (ds, sc) in EVENTS.items():
        hit = te.date.isin(pd.to_datetime(ds)).to_numpy()
        m = te[["station_id", "hour"]].merge(T[k.split("_")[0]], on=["station_id", "hour"], how="left").vn.fillna(0.0)
        out += sc * np.where(hit, m.to_numpy(), 0.0)
    return out


# ---------------- 5. 신규역 보정 ----------------
def newst_shift(P, alpha=0.5):
    h = tr[(tr.date > ANCHOR - pd.Timedelta(days=28))][["date", "station_id", "hour", "lb"]].copy()
    h["t"] = typ(h)
    L = h.groupby(["station_id", "t", "hour"]).lb.mean().rename("lvl").reset_index()
    p = te[K].assign(p=P)
    p["t"], p["m"] = typ(p), p.date.dt.to_period("M")
    p = p.merge(L, on=["station_id", "t", "hour"], how="left")
    p["d"] = p.p - p.lvl
    isnew = p.station_id.isin(NEW).to_numpy()
    D = p[~isnew].groupby(["m", "t", "hour"]).d.mean().rename("D").reset_index()
    q = p[isnew].merge(D, on=["m", "t", "hour"], how="left")
    q["gap"] = q.lvl + q.D - q.p
    sh = q.groupby(["station_id", "m", "t"]).gap.mean().rename("sh").reset_index()
    p = p.merge(sh, on=["station_id", "m", "t"], how="left")
    return np.where(isnew, P + alpha * p.sh.fillna(0.0).to_numpy(), P)


if __name__ == "__main__":
    import os
    LGB_TAGS = os.environ.get("FP_LGB", "lt1_full,lt2_full").split(",")
    OUT_NAME = os.environ.get("FP_OUT", "submission.csv")
    lgbm = np.mean([load_pred(t) for t in LGB_TAGS], axis=0)
    xgb = (load_pred("v3_x10") + load_pred("v3_x8")) / 2
    tree = newst_shift(0.75 * lgbm + 0.25 * xgb)
    chron = 0.5 * load_pred("chr_e3_full") + 0.5 * load_pred("wx_chr_full")
    P = 0.5 * tree + 0.5 * chron
    print("mean log: lgbm %.4f xgb %.4f chronos %.4f" % (lgbm.mean(), xgb.mean(), chron.mean()), flush=True)
    cr, ev = wx_correction(), game_correction()
    print(f"날씨 보정 평균 {cr.mean():+.4f} sd {cr.std():.4f} | 경기 보정 행 {(ev != 0).sum()}", flush=True)
    P = P + cr + 0.4 * ev
    htg = holiday_target()
    P = np.where(np.isnan(htg), P, 0.5 * P + 0.5 * np.nan_to_num(htg))
    print(f"명절 보정 행 {(~np.isnan(htg)).sum()}", flush=True)
    vn = venue_correction()
    P = np.clip(P + vn, 0, None)
    print(f"행사 보정 행 {(vn != 0).sum()}, 최대 {vn.max():.2f}", flush=True)
    sub = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(P)})
    assert len(sub) == 157194 and sub.boardings.notna().all() and (sub.boardings >= 0).all()
    sub.to_csv(WORK / OUT_NAME, index=False)
    print("->", WORK / OUT_NAME)
    print(sub.boardings.describe().round(1))
