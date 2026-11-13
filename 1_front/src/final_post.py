"""최종 앙상블 + 보정 -> submission.csv.   사용: python final_post.py <작업 폴더>  (testpred_*.csv, 보조 csv 가 있는 곳)

설정은 아래 FINAL 한 곳에 있다 (환경변수 FP_<키> 로 덮어쓰기 가능, 예: FP_WT=0.5). 순서:
  [혼합]
   1. 트리     = newst_shift( WL·LightGBM평균(lgb) + (1-WL)·XGBoost평균(xgb) )
   2. Chronos  = 0.5·chr_e3 + 0.5·wx_chr  (+ chr_hist·과거 7~10월 계절 보정)
   3. P        = WT·트리 + (1-WT)·Chronos
  [보정] 모두 기준일 6/30 이전 승차로만 학습, 7/1 이후는 메타데이터(날씨·경기·일정)만 사용
   4. 날씨(비·눈)             +1.0          wx_correction
   5. 경기 시작 시각 (SSG·부천) +0.4         game_correction
   6. 명절 목표 (추석 ±3일)     반반 섞기      holiday_target (holmatch: 평일/휴일 유형 맞춤)
   7. 신경망 (nn_mlp)         (1-nn)·P + nn·NN
   8. 신규역 추세             +newtrend     newst_trend
   9. 문학 경기 종료·관중       +ssgbox       ssgbox_correction
  10. 서울 버스 이동량 지수     +bus          bus_correction
  11. 공연·축제               +1.0          venue_correction
"""
import os
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

FINAL = dict(  # v8dh16 (out/sub_BASE.csv)
    lgb="lt1h_fullb,lt2h_fullb",                   # LightGBM 전체학습 2개 (날씨 포함)
    xgb="xh10_fullb,xh8_fullb,xh10_s2b,xh8_s3b",   # XGBoost 전체학습 4개
    wl=0.6,           # 트리 안 LightGBM 비중 (CV A·B)
    wt=0.55,          # 트리 vs Chronos (CV A·B)
    chrhist=0.25,     # Chronos 과거 계절 보정 비율
    holmatch=1,       # 명절 목표: 평일/휴일 유형 맞춘 과거일만
    nn="nn13b18b9_full:0.3",
    newtrend=0.5, ssgbox=0.5, bus=0.75,
    ghost_off=1,      # GHOSTIVAL(10/24~25) 취소 처리
    cancel_fix=1,     # 취소된 SSG 홈경기(8/5, 8/6, 8/22): 일정표 쓰는 모델의 경기 효과 제거
    chr="chr_e3_full,wx_chr_full",  # Chronos 2개 (같은 비중)
    lgb_new="", xgb_new="", chr_new="",  # 비어있지 않으면 신규역(3107~3109)만 이 모델들 사용 (2024 합성 학습 모델은 신규역에서 나쁨)
    out="submission.csv",
)
for _k in FINAL:  # FP_<KEY> 환경변수로 덮어쓰기
    _v = os.environ.get(f"FP_{_k.upper()}")
    if _v is not None:
        FINAL[_k] = type(FINAL[_k])(_v)


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


# ---------------- 4. 날씨 보정 ----------------
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


# ---------------- 5. 경기 보정 ----------------
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


# ---------------- 6. 명절 보정 (목표값) ----------------
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
        devs.append(x[["station_id", "hour", "rel", "t", "dev"]])
    # FP_HOLMATCH=1: 과거 명절일 중 평일/휴일 유형이 같은 날만 평균 (연휴 직전·직후 평일에 연휴 감소폭이 섞이지 않게)
    gk = ["station_id", "hour", "rel"] + (["t"] if FINAL["holmatch"] else [])
    D = pd.concat(devs).groupby(gk).dev.mean().rename("hdev").reset_index()
    d = te[K].copy()
    d["t"], d["rel"] = typ(d), rel_of(d.date)
    d = d.merge(D, on=gk, how="left").merge(
        lvl(h, ANCHOR).reset_index(), on=["station_id", "hour", "t"], how="left")
    tg = (d.lvl + d.hdev).to_numpy()
    return np.where((d.rel != 99).to_numpy() & ~np.isnan(tg), tg, np.nan)


# ---------------- 11. 공연·축제 보정 ----------------
VENUES = {"munhak": [3127], "asiad": [3209, 3210], "songdo": [3139], "bupyeong": [3119], "marathon": [3136]}
# 과거 행사일을 급증 감지 대신 공표 일정으로 지정하는 행사장 (부평풍물대축제 2025: 9/26~28, 부평대로 일대)
HIST_DAYS = {"bupyeong": ["2025-09-26", "2025-09-27", "2025-09-28"],
             # 인천대 송도캠퍼스 출발 마라톤: 2025 인천송도국제마라톤(9/28), 3.1절 마라톤(2025-03-01, 2026-03-01)
             "marathon": ["2025-03-01", "2025-09-28", "2026-03-01"]}
EVENTS = {  # 공표 일정 (웹): (행사장, 날짜들, 배율)  배율: 공연 1.0, 낮 페스티벌 1.0, 송도 축제 0.5 (공개 점수로 확인)
    # 부평풍물대축제(10/2~4)는 공개 점수에서 효과 없어 제외
    "munhak": (["2026-09-12", "2026-09-13"], 1.0),                                        # PLAVE 콘서트
    "munhak_fest": (["2026-08-29", "2026-08-30", "2026-10-24", "2026-10-25"], 1.0),        # Sustainable Wave, GHOSTIVAL
    "asiad": (["2026-07-11"], 1.0),                                                        # 싸이 흠뻑쇼
    "marathon": (["2026-09-20"], 1.0),                                                     # 인천송도국제마라톤 (경인일보, 9/20 08시 인천대 송도캠퍼스 출발)
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


# ---------------- 1·8. 신규역 보정 ----------------
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


def newst_trend(P):
    """신규역(3107~3109): 기존역 대비 평일 승차 비율이 최근 3개월 추세(월 최대 +0.03)를 이어가도록 맞추는 보정량.
    (3107 은 개통 직후 반짝 수요 뒤 7월 급감을 계절 패턴으로 오인해 7월 예측이 낮음)"""
    wd = lambda d: d[d.date.dt.dayofweek < 5]  # noqa: E731
    h = wd(tr).assign(ym=lambda d: d.date.dt.to_period("M"), g=lambda d: np.where(d.station_id.isin(NEW), d.station_id, 0))
    a = h.groupby(["ym", "g"]).boardings.sum().unstack() / h.groupby("ym").date.nunique().to_numpy()[:, None]
    rel = np.log(a[NEW].div(a[0], axis=0))
    t = wd(te.assign(b=np.expm1(P))).assign(ym=lambda d: d.date.dt.to_period("M"),
                                              g=lambda d: np.where(d.station_id.isin(NEW), d.station_id, 0))
    b = t.groupby(["ym", "g"]).b.sum().unstack() / t.groupby("ym").date.nunique().to_numpy()[:, None]
    relp = np.log(b[NEW].div(b[0], axis=0))
    c = np.zeros(len(te))
    for s_ in NEW:
        slope = float(np.clip(rel[s_].diff().iloc[-3:].mean(), 0.0, 0.03))
        for k, ym in enumerate(relp.index, start=1):
            tgt = rel[s_].iloc[-1] + slope * k
            m_ = (te.station_id == s_).to_numpy() & (te.date.dt.to_period("M") == ym).to_numpy()
            c[m_] = tgt - relp.loc[ym, s_]
        print(f"신규역 {s_}: 추세 {slope:+.3f}/월, 보정 {[round(float(tgt_), 3) for tgt_ in [rel[s_].iloc[-1] + slope * k - relp.iloc[k - 1][s_] for k in range(1, len(relp) + 1)]]}", flush=True)
    return c


# ---------------- 9. 문학 SSG 경기 종료 시각·관중 보정 ----------------
# CV(A·B) 잔차에서 맞춘 값: 종료 시각이 속한 시간 기준 상대시간 k 별 (편향, log 관중 기울기). A로 맞춰 B, B로 맞춰 A 모두 개선(x0.5: -0.0005/-0.0007)
SSG_BOX = {-3: (0.193, 0.081), -2: (0.263, -0.52), -1: (-0.715, -1.529), 0: (-0.427, -2.459), 1: (0.45, -1.043), 2: (0.863, -0.237)}
SSG_LC = 9.7408


def ssgbox_correction():
    """반환: 행별 보정값(더할 값). geo/ssg_box.csv: KBO GameCenter 관중 수·종료 시각."""
    b = pd.read_csv(V._find("ssg_box.csv"), parse_dates=["date"])
    b["endh"] = b.end.str[:2].astype(int) + b.end.str[3:5].astype(int) / 60
    d = te[K].merge(b[["date", "endh", "crowd"]], on="date", how="left")
    k = np.floor(d.hour - d.endh)
    c = np.zeros(len(d))
    for kk, (a, s) in SSG_BOX.items():
        m = (d.station_id == 3127).to_numpy() & (k == kk).to_numpy()
        c[m] = -(a + s * (np.log(d.crowd.to_numpy()[m]) - SSG_LC))
    return c


# ---------------- 10. 서울 버스 일별 이동량 지수 보정 ----------------
# geo/seoul_bus_index.csv: 서울 열린데이터광장 OA-12912 (버스 정류장별 일별 승하차) 서울 전체 승차 log 합계의
# 같은 요일유형 앞뒤 8주 중앙값 대비 차이 (src/bus_day.py). CV 두 구간 합친 기울기 0.7165, A->B / B->A 모두 개선.
BUS_BETA = 0.7165


def bus_correction():
    b = pd.read_csv(V._find("seoul_bus_index.csv"), parse_dates=["date"]).set_index("date").bus_idx
    return BUS_BETA * te.date.map(b).fillna(0.0).clip(-0.5, 0.5).to_numpy()


# ---------------- 12. 취소된 행사·경기: 문학(3127) 예측을 가까운 같은 요일(±4주 안, 경기·행사·공휴일 없는 날 최대 4개) 평균으로 ----------------
GHOST_DAYS = ["2026-10-24", "2026-10-25"]
CANCEL_DAYS = ["2026-08-05", "2026-08-06", "2026-08-22"]


def neighbour_fill(P, days):
    busy = set(pd.read_csv(V._find("ssg_home_actual.csv"), parse_dates=["date"]).date)
    for k, (ds, _) in EVENTS.items():
        busy |= set(pd.to_datetime(ds))
    busy |= set(pd.to_datetime(GHOST_DAYS + CANCEL_DAYS))
    P = P.copy()
    d3127 = (te.station_id == 3127).to_numpy()
    for d in pd.to_datetime(days):
        nb = [d + pd.Timedelta(days=7 * k) for k in (-1, 1, -2, 2, -3, 3, -4, 4)]  # 가까운 순, 최대 4개
        nb = [x for x in nb if x not in busy and x not in V.HSET and te.date.min() <= x <= te.date.max()][:4]
        if len(nb) < 2:  # 이웃이 하나뿐이면 계절 차이가 커서 그대로 둠
            print(f"  12 문학 {d.date()}: 이웃 {len(nb)}개 -> 유지", flush=True)
            continue
        tgt = d3127 & (te.date == d).to_numpy()
        ref = pd.DataFrame({"hour": te.hour[d3127 & te.date.isin(nb).to_numpy()], "p": P[d3127 & te.date.isin(nb).to_numpy()]}).groupby("hour").p.mean()
        P[tgt] = te.hour[tgt].map(ref).to_numpy()
        print(f"  12 문학 {d.date()} -> 이웃 {[x.strftime('%m-%d') for x in nb]} 평균", flush=True)
    return P


def report(name, c):
    print(f"  {name:<14} 행 {(c != 0).sum():>7}  평균 {c.mean():+.4f}  범위 {c.min():+.3f}~{c.max():+.3f}", flush=True)


if __name__ == "__main__":
    F = FINAL
    print("설정", F, flush=True)
    # [혼합] 1~3
    isnew = te.station_id.isin(NEW).to_numpy()
    avg = lambda tags: np.mean([load_pred(t) for t in tags.split(",")], axis=0)  # noqa: E731
    lgbm, xgb = avg(F["lgb"]), avg(F["xgb"])
    if F["lgb_new"]:
        lgbm = np.where(isnew, avg(F["lgb_new"]), lgbm)
    if F["xgb_new"]:
        xgb = np.where(isnew, avg(F["xgb_new"]), xgb)
    tree = newst_shift(F["wl"] * lgbm + (1 - F["wl"]) * xgb)
    chron = avg(F["chr"])
    if F["chr_new"]:
        chron = np.where(isnew, avg(F["chr_new"]), chron)
    if F["chrhist"]:  # 과거(2016~2024) 평균 7~10월 계절 변화와 Chronos 계절 변화 차이를 chrhist 만큼 메움
        import hist_season as HS
        dh = HS.hist_delta(HS.hist_table(meta), [2016, 2017, 2018, 2019, 2022, 2023, 2024], 0.85)
        chron = chron + HS.correction(te[K].assign(P=chron), tr, ANCHOR, dh, F["chrhist"], "sh")
    P = F["wt"] * tree + (1 - F["wt"]) * chron
    print(f"평균 log: lgb {lgbm.mean():.4f} xgb {xgb.mean():.4f} chronos {chron.mean():.4f} -> P {P.mean():.4f}", flush=True)
    # [보정] 4~11
    c = wx_correction(); report("4 날씨", c); P = P + c
    c = 0.4 * game_correction(); report("5 경기 시작", c); P = P + c
    htg = holiday_target()
    P = np.where(np.isnan(htg), P, 0.5 * P + 0.5 * np.nan_to_num(htg)); print(f"  6 명절 목표     행 {(~np.isnan(htg)).sum():>7}", flush=True)
    if F["nn"]:
        tag, w = F["nn"].split(":")
        P = (1 - float(w)) * P + float(w) * np.clip(load_pred(tag), 0, None); print(f"  7 신경망 {tag} x{w}", flush=True)
    if F["newtrend"]:
        c = F["newtrend"] * newst_trend(P); report("8 신규역 추세", c); P = P + c
    if F["ssgbox"]:
        c = F["ssgbox"] * ssgbox_correction(); report("9 문학 종료·관중", c); P = P + c
    if F["bus"]:
        c = F["bus"] * bus_correction(); report("10 서울 버스", c); P = P + c
    if F["ghost_off"]:
        EVENTS["munhak_fest"] = ([x for x in EVENTS["munhak_fest"][0] if x not in GHOST_DAYS], EVENTS["munhak_fest"][1])
    c = venue_correction(); report("11 공연·축제", c)
    P = np.clip(P + c, 0, None)
    fill = (GHOST_DAYS if F["ghost_off"] else []) + (CANCEL_DAYS if F["cancel_fix"] else [])
    if fill:
        P = neighbour_fill(P, fill)
    sub = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(P)})
    assert len(sub) == 157194 and sub.boardings.notna().all() and (sub.boardings >= 0).all()
    sub.to_csv(WORK / F["out"], index=False)
    print("->", WORK / F["out"])
    print(sub.boardings.describe().round(1))
