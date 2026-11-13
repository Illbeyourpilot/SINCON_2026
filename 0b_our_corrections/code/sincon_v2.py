"""SINCON 2026 v2 파이프라인 (단일 파일, Kaggle/로컬 공용).

CONFIG 는 커널 생성 시 파일 맨 위 CONFIG_OVERRIDE 로 덮어쓴다.
  mode      : "cv" (검증 폴드 점수 + 예측 저장) | "full" (6/30 기준 학습, 제출 파일)
  model     : "lgb" (CPU) | "xgb" (GPU 있으면 cuda)
  groups    : 사용할 피처 그룹 목록
  folds     : cv 에서 돌릴 폴드
  seeds, rounds, anchor_step
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = dict(mode="cv", model="lgb", groups=["base", "newst", "graph", "analog"], folds=["A", "B", "C"],
              seeds=[0], rounds=1000, lr=0.05, anchor_step=7, cs=True, tag="v2")
CONFIG.update(globals().get("CONFIG_OVERRIDE", {}))

ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
try:
    _ROOT = Path(__file__).resolve().parent.parent
except NameError:  # 노트북/exec 실행
    _ROOT = Path.cwd()
# 탐색 순서: SINCON_DIRS(콜론 구분) -> 작업 폴더 -> /kaggle/input -> 로컬 프로젝트 data/geo
_DIRS = [Path(x) for x in os.environ.get("SINCON_DIRS", "").split(":") if x] + [Path.cwd()]
if ON_KAGGLE:
    _DIRS += [Path("/kaggle/working")]
_DIRS += [_ROOT / "data", _ROOT / "geo"]


def _find(name):
    for d in _DIRS:
        if (d / name).exists():
            return d / name
    if Path("/kaggle/input").exists():
        hits = sorted(Path("/kaggle/input").glob(f"**/{name}"))
        if hits:
            return hits[0]
    return None


OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("SINCON_OUT", _ROOT / "out"))
OUT.mkdir(exist_ok=True)
LOG = open(OUT / f"log_{CONFIG['tag']}.txt", "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


HORIZON = 123
HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",
    "2025-03-01", "2025-03-03", "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06",
    "2025-08-15", "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
    "2025-10-09", "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01", "2026-03-02",
    "2026-05-01", "2026-05-05", "2026-05-24", "2026-05-25", "2026-06-03", "2026-06-06",
    "2026-07-17", "2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26",
    "2026-10-03", "2026-10-05", "2026-10-09",
])
SOFT_HOLIDAYS = pd.to_datetime(["2025-05-01"])
BIG_HOLIDAYS = pd.to_datetime(["2025-01-29", "2025-10-06", "2026-02-17", "2026-09-25"])
# 방학 (연초 공표 학사일정 기준 근사). 대학: 인천대·인하대·연세대 국제캠 / 초중고: 인천 일반 학교
UNIV_VAC = [("2024-12-21", "2025-03-02"), ("2025-06-21", "2025-08-31"), ("2025-12-20", "2026-03-01"),
            ("2026-06-20", "2026-08-31"), ("2026-12-19", "2027-02-28")]
SCHOOL_VAC = [("2024-12-28", "2025-03-03"), ("2025-07-25", "2025-08-17"), ("2026-01-02", "2026-03-02"),
              ("2026-07-24", "2026-08-16"), ("2027-01-01", "2027-03-01")]
UNIV_START = pd.to_datetime(["2024-09-02", "2025-03-03", "2025-09-01", "2026-03-02", "2026-09-01"])
HSET = set(HOLIDAYS) | set(SOFT_HOLIDAYS)
INT_COLS = {"station_id", "line", "hour", "dow", "daytype", "month", "dom", "doy", "woy", "horizon", "hol_prev",
            "hol_next", "bridge", "after_long_off", "big_hol_rel", "is_new", "univ_vac", "school_vac"}
CAT_COLS = ("station_id", "line")
FOLDS = {"A": ("2025-06-30", "2025-10-31"), "B": ("2026-02-28", "2026-06-30"), "C": ("2025-10-31", "2026-02-28")}


def load():
    train = pd.read_csv(_find("train.csv"), parse_dates=["date"])
    test = pd.read_csv(_find("test.csv"), parse_dates=["date"])
    meta = pd.read_csv(_find("metaData.csv"), parse_dates=["first_date"])
    geo = pd.read_csv(_find("station_geo_graph.csv"), encoding="utf-8-sig")
    train["lb"] = np.log1p(train["boardings"]).astype("float32")
    train["la"] = np.log1p(train["alightings"]).astype("float32")
    for df in (train, test):
        df["dow"] = df.date.dt.dayofweek
        df["daytype"] = np.where(df.date.isin(HSET), 2, np.where(df.dow >= 5, 1, 0))
    return train, test, meta, geo


def calendar_feats(df):
    d = df.date
    df["month"] = d.dt.month
    df["dom"] = d.dt.day
    df["doy"] = d.dt.dayofyear
    df["woy"] = d.dt.isocalendar().week.astype(int).to_numpy()
    prev_, next_ = d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)
    df["hol_prev"] = prev_.isin(HSET).astype(np.int8)
    df["hol_next"] = next_.isin(HSET).astype(np.int8)
    off_prev = prev_.isin(HSET) | (prev_.dt.dayofweek >= 5)
    off_next = next_.isin(HSET) | (next_.dt.dayofweek >= 5)
    df["bridge"] = ((df.daytype == 0) & off_prev & off_next).astype(np.int8)
    df["after_long_off"] = ((df.daytype == 0) & prev_.isin(HSET)).astype(np.int8)
    rel = np.full(len(df), 99, dtype=np.int16)
    for b in BIG_HOLIDAYS:
        r = (d - b).dt.days.to_numpy()
        m = np.abs(r) <= 10
        rel[m] = r[m]
    df["big_hol_rel"] = rel
    for nm, per in (("univ_vac", UNIV_VAC), ("school_vac", SCHOOL_VAC)):
        m = np.zeros(len(df), dtype=np.int8)
        for a_, b_ in per:
            m |= ((d >= a_) & (d <= b_)).to_numpy()
        df[nm] = m
    dn = d.to_numpy()
    last = np.full(len(df), np.nan)
    for st in UNIV_START:
        k = (dn - st.to_datetime64()) / np.timedelta64(1, "D")
        last = np.where((k >= 0) & (np.isnan(last) | (k < last)), k, last)
    df["univ_term_day"] = np.minimum(last, 60)
    return df


class Builder:
    """[날짜 x 역 x 시간] 텐서 + 날짜 누적합으로 모든 창 통계를 O(1) 로 계산하는 벡터화 빌더."""

    def __init__(self, train, meta, geo, groups):
        self.groups = set(groups)
        self.d0 = train.date.min()
        self.dates = pd.date_range(self.d0, train.date.max())
        self.sids = np.array(sorted(meta.station_id))
        self.sidx = {s: i for i, s in enumerate(self.sids)}
        D, S, H = len(self.dates), len(self.sids), 18
        di = (train.date - self.d0).dt.days.to_numpy()
        si = train.station_id.map(self.sidx).to_numpy()
        hi = train.hour.to_numpy() - 6
        self.LB = np.full((D, S, H), np.nan, np.float64)
        self.LA = np.full((D, S, H), np.nan, np.float64)
        self.LB[di, si, hi] = train.lb.to_numpy()
        self.LA[di, si, hi] = train.la.to_numpy()
        self.dt = np.where(self.dates.isin(HSET), 2, np.where(self.dates.dayofweek >= 5, 1, 0))
        self.dow = self.dates.dayofweek.to_numpy()
        ok = ~np.isnan(self.LB)
        lb0 = np.where(ok, self.LB, 0.0)
        la0 = np.where(ok, self.LA, 0.0)

        def cum(x):
            c = np.zeros((D + 1,) + x.shape[1:])
            np.cumsum(x, axis=0, out=c[1:])
            return c

        def sel(m):
            m = m[:, None, None]
            return cum(np.where(m, lb0, 0)), cum(np.where(m, ok, 0)), cum(np.where(m, lb0 ** 2, 0))
        self.C_dt = [sel(self.dt == k) for k in (0, 1, 2)]
        self.C_dow = [sel((self.dow == j) & (self.dt < 2)) for j in range(7)]
        self.C_la = [cum(np.where((self.dt == k)[:, None, None], la0, 0)) for k in (0, 1)]
        self.G = np.nanmean(self.LB.reshape(D, -1), axis=1)
        self.fd = meta.set_index("station_id").first_date.reindex(self.sids).to_numpy()
        self.line = meta.set_index("station_id").canonical_line.reindex(self.sids).to_numpy()
        if "nbr" in self.groups:
            _, A_ = adjacency(meta, geo)  # 노선 + 1.5km 근접, sids 와 같은 정렬 순서
            self.W = A_ / np.maximum(A_.sum(1, keepdims=True), 1)
            self.W1 = (A_ + np.eye(len(A_))) / (A_.sum(1, keepdims=True) + 1)  # 자기 포함 평활용
        self.cats = {"station_id": [int(x) for x in self.sids], "line": sorted(int(x) for x in set(self.line))}
        self.g_feats = [c for c in geo.columns if c != "station_id" and not c.startswith("g2_")]
        self.G_arr = geo.set_index("station_id").reindex(self.sids)[self.g_feats].to_numpy(np.float32)
        if "event" in self.groups:
            # SSG 문학 홈경기: 시즌 전 발표된 원 일정만 (잔여경기·취소 여부는 사후 정보라 제외)
            sch = pd.read_csv(_find(ev_file("ssg")), parse_dates=["date"])
            self.ssg_start = sch.set_index("date").start_hour
            self.munhak_w = np.exp(-geo.set_index("station_id").reindex(self.sids).dist_munhak.to_numpy() / 1000.0)
            # 부천FC 부천종합운동장 홈경기 (시즌 전 공표분: K리그1 2026 은 스플릿 전 33라운드까지)
            bfc = pd.read_csv(_find(ev_file("bucheon")), parse_dates=["date"])
            self.bfc_start = bfc.set_index("date").start_hour
            self.bfc_w = np.exp(-geo.set_index("station_id").reindex(self.sids).dist_bucheon_stadium.to_numpy() / 1000.0)

    @staticmethod
    def _win(C, lo, hi):
        """누적합 C 에서 날짜 인덱스 (lo, hi] 구간 합."""
        lo = max(lo, -1)
        return C[hi + 1] - C[lo + 1]

    def _mean(self, sel, lo, hi):
        s, n, _ = sel
        n_ = self._win(n, lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_ > 0, self._win(s, lo, hi) / np.maximum(n_, 1), np.nan)

    def _std(self, sel, lo, hi):
        s, n, q = sel
        s_, n_, q_ = self._win(s, lo, hi), self._win(n, lo, hi), self._win(q, lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (q_ - s_ ** 2 / np.maximum(n_, 1)) / np.maximum(n_ - 1, 1)
            return np.where(n_ > 1, np.sqrt(np.maximum(v, 0)), np.nan)

    def _gather(self, idx, ss, hh, ok):
        v = np.full(len(idx), np.nan)
        v[ok] = self.LB[idx[ok], ss[ok], hh[ok]]
        return v

    def rows(self, anchor, end=None, target=None):
        anchor = pd.Timestamp(anchor)
        a = (anchor - self.d0).days
        if target is None:
            hi_ = min((pd.Timestamp(end) - self.d0).days, len(self.dates) - 1)
            if hi_ <= a:
                return pd.DataFrame()
            sub = self.LB[a + 1:hi_ + 1]
            dd, ss, hh = np.nonzero(~np.isnan(sub))
            di = dd + a + 1
            out = pd.DataFrame({"date": self.dates[di], "station_id": self.sids[ss], "hour": hh + 6,
                                "lb": sub[dd, ss, hh].astype(np.float32)})
        else:
            out = target.reset_index(drop=True)[["example_id", "date", "station_id", "hour"]].copy()
            di = (out.date - self.d0).dt.days.to_numpy()
            ss = out.station_id.map(self.sidx).to_numpy()
            hh = out.hour.to_numpy() - 6
        dates = out.date
        dow = dates.dt.dayofweek.to_numpy()
        dtp = np.where(dates.isin(HSET), 2, np.where(dow >= 5, 1, 0))
        out["dow"], out["daytype"] = dow, dtp
        k01 = np.where(dtp < 2, dtp, 0)
        nanhol = np.where(dtp < 2, 1.0, np.nan)
        F = {}
        for w, days in (("4w", 28), ("8w", 56), ("26w", 182), ("all", a + 1)):
            M = np.stack([self._mean(self.C_dt[k], a - days, a) for k in (0, 1)])
            F[f"p_dt_{w}"] = M[k01, ss, hh] * nanhol
            if w != "4w":
                Mw = np.stack([self._mean(self.C_dow[j], a - days, a) for j in range(7)])
                F[f"p_dow_{w}"] = Mw[dow, ss, hh]
        F["p_hol_all"] = self._mean(self.C_dt[2], -1, a)[ss, hh]
        F["p_sun_all"] = self._mean(self.C_dow[6], -1, a)[ss, hh]
        Sd = np.stack([self._std(self.C_dow[j], a - 56, a) for j in range(7)])
        F["sd_dow_8w"] = Sd[dow, ss, hh]
        La = []
        for k in (0, 1):
            n_ = self._win(self.C_dt[k][1], a - 56, a)
            with np.errstate(invalid="ignore", divide="ignore"):
                La.append(np.where(n_ > 0, self._win(self.C_la[k], a - 56, a) / np.maximum(n_, 1), np.nan))
        F["la_dt_8w"] = np.stack(La)[k01, ss, hh] * nanhol

        def st_level(days):
            s_ = sum(self._win(self.C_dt[k][0], a - days, a).sum(1) for k in (0, 1))
            n_ = sum(self._win(self.C_dt[k][1], a - days, a).sum(1) for k in (0, 1))
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
        F["st_level_4w"] = st_level(28)[ss]
        F["st_level_26w"] = st_level(182)[ss]
        F["st_level_all"] = st_level(a + 1)[ss]
        F["trend_4_26"] = F["st_level_4w"] - F["st_level_26w"]
        F["trend_26_all"] = F["st_level_26w"] - F["st_level_all"]
        if "trend" in self.groups:
            # 평일만으로 수준을 잰다 (창마다 주말·공휴일 비중이 달라 생기는 왜곡 제거)
            def wd_lvl(days, end):
                if end - days < -1:
                    return np.full(len(self.sids), np.nan)
                s_ = self._win(self.C_dt[0][0], end - days, end).sum(1)
                n_ = self._win(self.C_dt[0][1], end - days, end).sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(n_ >= 18 * 10, s_ / np.maximum(n_, 1), np.nan)
            l4, l13 = wd_lvl(28, a), wd_lvl(91, a)
            yo = wd_lvl(56, a) - wd_lvl(56, a - 364)
            F["wd_tr_4_13"] = (l4 - l13)[ss]
            F["wd_yoy_8w"] = yo[ss]
            with np.errstate(invalid="ignore"):
                F["net_tr_4_13"] = np.full(len(out), np.nanmean(l4 - l13) if np.isfinite(l4 - l13).any() else np.nan)
                F["net_yoy_8w"] = np.full(len(out), np.nanmean(yo) if np.isfinite(yo).any() else np.nan)
            if "nbr" in self.groups:
                def nb(v):
                    m = np.isfinite(v)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        return (self.W @ np.where(m, v, 0)) / (self.W @ m)
                F["nbr_tr_4_13"] = nb(l4 - l13)[ss]
                F["nbr_yoy_8w"] = nb(yo)[ss]
        F["hol_minus_sun"] = F["p_hol_all"] - F["p_sun_all"]

        yd = di - 364
        ya = a - 364
        yd_ok = (yd >= 0) & (yd <= a)
        ly = self._gather(yd, ss, hh, yd_ok)
        if ya - 28 >= 0:
            Bm = np.stack([self._mean(self.C_dt[k], ya - 28, ya) for k in (0, 1)])
            F["ya_seas_st"] = ly - Bm[k01, ss, hh] * nanhol
            ydt = np.where(yd_ok, self.dt[np.clip(yd, 0, len(self.dt) - 1)], -1)
            ball = []
            for k in (0, 1):
                s_ = self._win(self.C_dt[k][0], ya - 28, ya).sum()
                n_ = self._win(self.C_dt[k][1], ya - 28, ya).sum()
                ball.append(s_ / n_ if n_ > 0 else np.nan)
            ball = np.array(ball + [np.nan])
            F["ya_daytype"] = np.where(ydt >= 0, ydt, np.nan)
            gy = np.where(yd_ok, self.G[np.clip(yd, 0, len(self.G) - 1)], np.nan)
            F["ya_seas_all"] = np.where(ydt >= 0, gy - ball[np.clip(ydt, 0, 2)], np.nan)
        else:
            F["ya_seas_st"] = np.full(len(out), np.nan)
            F["ya_daytype"] = np.full(len(out), np.nan)
            F["ya_seas_all"] = np.full(len(out), np.nan)

        F["horizon"] = di - a
        F["line"] = self.line[ss]
        fd = pd.to_datetime(self.fd[ss])
        isnew = np.asarray(fd > pd.Timestamp("2025-01-01"))
        F["is_new"] = isnew.astype(np.int8)
        if "newst" in self.groups:
            F["days_open_new"] = np.where(isnew, (dates.to_numpy() - fd.to_numpy()) / np.timedelta64(1, "D"), np.nan)
        if "analog" in self.groups:
            # 절대 수준이 아니라 '그해 직전 4주 대비 변화'만 넣는다 (수준을 넣으면 연간 성장분만큼 과소예측)
            base_ya = (np.stack([self._mean(self.C_dt[k], ya - 28, ya) for k in (0, 1)])[k01, ss, hh]
                       if ya - 28 >= 0 else np.full(len(out), np.nan))
            acc = np.where(np.isnan(ly), 0.0, ly)
            cnt = (~np.isnan(ly)).astype(float)
            for k in (-7, 7):
                y2 = yd + k
                v = self._gather(y2, ss, hh, (y2 >= 0) & (y2 <= a))
                g = ~np.isnan(v)
                acc[g] += v[g]
                cnt[g] += 1
            F["ly_wk_rel"] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan) - base_ya
        if "nbr" in self.groups:
            # 작년 계절 편차를 이웃 역(자기 포함) x 앞뒤 1시간으로 평활 -> 역x시간 단위 잡음 감소
            v = np.asarray(F["ya_seas_st"], dtype=float)
            d0i = di - di.min()
            Y = np.full((d0i.max() + 1, len(self.sids), 18), np.nan)
            Y[d0i, ss, hh] = v
            Mk = np.isfinite(Y).astype(float)
            Y0 = np.where(Mk > 0, Y, 0.0)
            Ys, Ms = Y0.copy(), Mk.copy()
            Ys[:, :, 1:] += Y0[:, :, :-1]
            Ys[:, :, :-1] += Y0[:, :, 1:]
            Ms[:, :, 1:] += Mk[:, :, :-1]
            Ms[:, :, :-1] += Mk[:, :, 1:]
            num = np.einsum("ij,djh->dih", self.W1, Ys)
            den = np.einsum("ij,djh->dih", self.W1, Ms)
            with np.errstate(invalid="ignore", divide="ignore"):
                F["ya_seas_sm"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)[d0i, ss, hh]
        out = pd.concat([out, pd.DataFrame(F)], axis=1)
        out = calendar_feats(out)
        if "analog" in self.groups:
            rel = out.big_hol_rel.to_numpy().astype(int)
            acc = np.zeros(len(out))
            cnt = np.zeros(len(out))
            for b in BIG_HOLIDAYS:
                b0 = (b - self.d0).days
                if b0 - 11 - 28 < 0 or b0 - 11 > a:  # 기준 구간이 anchor 이전에 관측돼야 함 (누출 방지)
                    continue
                bi = b0 + np.clip(rel, -10, 10)
                v = self._gather(bi, ss, hh, (rel != 99) & (bi >= 0) & (bi <= a))
                # 해당 명절 직전(-11일 이전) 4주 같은 요일유형 평균 대비
                base_b = np.stack([self._mean(self.C_dt[k], b0 - 11 - 28, b0 - 11) for k in (0, 1)])[k01, ss, hh]
                v = v - base_b
                g = ~np.isnan(v)
                acc[g] += v[g]
                cnt[g] += 1
            out["analog_bighol_rel"] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        if "event" in self.groups:
            st = out.date.map(self.ssg_start).to_numpy(dtype=float)
            game = ~np.isnan(st)
            out["ssg_hr_rel"] = np.where(game, out.hour.to_numpy() - st, np.nan)
            out["ssg_expo"] = np.where(game, self.munhak_w[ss], 0.0)
            bst = out.date.map(self.bfc_start).to_numpy(dtype=float)
            bg = ~np.isnan(bst)
            out["bfc_hr_rel"] = np.where(bg, out.hour.to_numpy() - bst, np.nan)
            out["bfc_expo"] = np.where(bg, self.bfc_w[ss], 0.0)
        if "graph" in self.groups:
            out = pd.concat([out, pd.DataFrame(self.G_arr[ss], columns=self.g_feats)], axis=1)
        if "wx" in self.groups:
            out = pd.concat([out, weather_feats(out)], axis=1)
        for c in out.columns:
            if c in ("date", "example_id"):
                continue
            out[c] = out[c].astype(np.int32 if c in INT_COLS else np.float32)
        return out

    def feats(self):
        f = ["station_id", "line", "hour", "dow", "daytype", "month", "dom", "doy", "woy", "horizon",
             "hol_prev", "hol_next", "bridge", "after_long_off", "big_hol_rel",
             "p_dt_4w", "p_dt_8w", "p_dt_26w", "p_dow_8w", "p_dow_26w", "p_dt_all", "p_dow_all", "p_hol_all",
             "p_sun_all", "sd_dow_8w", "la_dt_8w", "st_level_4w", "st_level_26w", "st_level_all", "trend_4_26",
             "trend_26_all", "hol_minus_sun", "ya_seas_st", "ya_seas_all", "ya_daytype", "is_new"]
        if "newst" in self.groups:
            f.append("days_open_new")
        if "vac" in self.groups:
            f += ["univ_vac", "school_vac", "univ_term_day"]
        if "trend" in self.groups:
            f += ["wd_tr_4_13", "wd_yoy_8w", "net_tr_4_13", "net_yoy_8w"]
            if "nbr" in self.groups:
                f += ["nbr_tr_4_13", "nbr_yoy_8w"]
        if "nbr" in self.groups:
            f += ["ya_seas_sm"]
        if "analog" in self.groups:
            f += ["ly_wk_rel", "analog_bighol_rel"]
        if "event" in self.groups:
            f += ["ssg_hr_rel", "ssg_expo"] + (["bfc_hr_rel", "bfc_expo"] if CONFIG.get("event_bfc", True) else [])
        if "graph" in self.groups:
            f += self.g_feats
        if "wx" in self.groups:
            f += list(WX_COLS)
        return f

    def train_set(self, last_target):
        last = pd.Timestamp(last_target)
        anchors = pd.date_range("2025-02-26", last - pd.Timedelta(days=7), freq=f"{CONFIG['anchor_step']}D")
        if CONFIG.get("anchor_align"):
            # 학습 anchor 를 예측 기준일(last)과 같은 요일로 맞춤 (horizon↔요일 관계 일치)
            step = CONFIG["anchor_step"]
            anchors = pd.DatetimeIndex(sorted(last - pd.Timedelta(days=7 + step * k) for k in range(400)
                                              if last - pd.Timedelta(days=7 + step * k) >= pd.Timestamp("2025-02-26")))
        if CONFIG.get("smoke"):
            anchors = anchors[-4:]
        parts = [self.rows(a, min(a + pd.Timedelta(days=HORIZON), last)) for a in anchors]
        return pd.concat(parts, ignore_index=True), len(anchors)


class Tick:
    """단계별 경과 시간 기록."""

    def __init__(self, name):
        self.name, self.t0, self.t = name, time.time(), time.time()

    def __call__(self, msg):
        now = time.time()
        p(f"  [{self.name}] {msg}  (+{now - self.t:.1f}s, 누적 {now - self.t0:.0f}s)")
        self.t = now


def gpu_available():
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def norm_base(df):
    """정규화 기준: 최근 8주 같은 요일 평균 (없으면 26주 -> 전체 -> 역 수준)."""
    b = df["p_dow_8w"].astype(float)
    for c in ("p_dow_26w", "p_dow_all", "st_level_4w"):
        b = b.fillna(df[c].astype(float))
    return b.to_numpy()


def fit_predict(tr, X_list, feats, seeds, rounds, cats):
    if CONFIG.get("norm"):
        bt = norm_base(tr)
        fill = float(np.nanmean(tr.lb.to_numpy() - bt) + np.nanmean(bt))
        bt = np.where(np.isnan(bt), fill, bt)
        tr = tr.assign(lb=tr.lb.to_numpy() - bt)
        bxs = [np.where(np.isnan(b_), fill, b_) for b_ in (norm_base(x) for x in X_list)]
        outs = _fit_predict(tr, X_list, feats, seeds, rounds, cats, clip=False)
        return [np.clip(o + b_, 0, None) for o, b_ in zip(outs, bxs)]
    return _fit_predict(tr, X_list, feats, seeds, rounds, cats)


def _fit_predict(tr, X_list, feats, seeds, rounds, cats, clip=True):
    """X_list 의 각 프레임에 대한 예측(log 공간) 목록. cats: 범주형 열의 고정 범주 목록."""
    outs = [np.zeros(len(x)) for x in X_list]
    tk = Tick(f"fit {CONFIG['model']} n={len(tr):,}")
    if CONFIG["model"] == "xgb":
        import xgboost as xgb
        dev = "cuda" if gpu_available() else "cpu"
        if ON_KAGGLE and CONFIG.get("require_gpu", True) and dev != "cuda":
            raise RuntimeError("GPU 커널인데 GPU 를 찾지 못함")

        def prep(df):
            X = df[feats].copy()
            for c in CAT_COLS:
                X[c] = pd.Categorical(X[c], categories=cats[c])
                assert X[c].notna().all(), f"{c} 에 범주 목록 밖 값"
            return X
        dtr = xgb.QuantileDMatrix(prep(tr), tr.lb, enable_categorical=True, max_bin=256)
        dxs = [xgb.QuantileDMatrix(prep(x), ref=dtr, enable_categorical=True) for x in X_list]
        tk(f"DMatrix 생성 device={dev}")

        class Prog(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if (epoch + 1) % 200 == 0:
                    tk(f"round {epoch + 1}/{rounds}")
                return False
        for sd in seeds:
            prm = dict(objective="reg:squarederror", eta=CONFIG["lr"], max_depth=10, min_child_weight=50,
                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", device=dev,
                       max_cat_to_onehot=1, max_bin=256, seed=sd)
            prm.update(CONFIG.get("xgb_params", {}))
            m = xgb.train(prm, dtr, rounds, callbacks=[Prog()])
            for i, dx in enumerate(dxs):
                outs[i] += m.predict(dx) / len(seeds)
            tk(f"seed {sd} 학습+예측 완료")
        p(f"    xgb device={dev} rounds={rounds} seeds={len(seeds)}")
    elif CONFIG["model"] == "cat":
        from catboost import CatBoostRegressor, Pool
        dev = "GPU" if gpu_available() else "CPU"

        def prep(df):
            X = df[feats].copy()
            for c in CAT_COLS:
                X[c] = X[c].astype(int)
            return X
        ci = [feats.index(c) for c in CAT_COLS if c in feats]
        ptr = Pool(prep(tr), tr.lb.to_numpy(), cat_features=ci)
        pxs = [Pool(prep(x), cat_features=ci) for x in X_list]
        tk(f"Pool 생성 device={dev}")
        for sd in seeds:
            prm = dict(loss_function="RMSE", iterations=rounds, learning_rate=CONFIG["lr"], depth=8,
                       l2_leaf_reg=3.0, border_count=254, task_type=dev, random_seed=sd, verbose=200,
                       bootstrap_type="Bernoulli", subsample=0.8)
            prm.update(CONFIG.get("cat_params", {}))
            m = CatBoostRegressor(**prm)
            m.fit(ptr)
            for i, px in enumerate(pxs):
                outs[i] += m.predict(px) / len(seeds)
            tk(f"seed {sd} 학습+예측 완료")
        del ptr
    else:
        import lightgbm as lgb

        def prep(df):
            X = df[feats].copy()
            for c in CAT_COLS:
                X[c] = pd.Categorical(X[c], categories=cats[c])
                assert X[c].notna().all(), f"{c} 에 범주 목록 밖 값"
            return X
        ds = lgb.Dataset(prep(tr), tr.lb, categorical_feature=list(CAT_COLS), free_raw_data=False)
        Xs = [prep(x) for x in X_list]
        for sd in seeds:
            prm = dict(objective="regression", learning_rate=CONFIG["lr"], num_leaves=127, min_data_in_leaf=200,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                       num_threads=0, verbose=-1, seed=sd)
            prm.update(CONFIG.get("lgb_params", {}))
            m = lgb.train(prm, ds, rounds, callbacks=[lambda env: (env.iteration + 1) % 200 == 0 and tk(
                f"round {env.iteration + 1}/{rounds}")])
            for i, x in enumerate(Xs):
                outs[i] += m.predict(x) / len(seeds)
            tk(f"seed {sd} 학습+예측 완료")
    return [np.clip(o, 0, None) for o in outs] if clip else outs


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(h))


def adjacency(meta, geo):
    ids = sorted(meta.station_id)
    idx = {s: i for i, s in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)))
    for ln in (1, 2, 7):
        seq = sorted(meta[meta.canonical_line == ln].station_id)
        for a, b in zip(seq, seq[1:]):
            A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1
    for a, b in [(3124, 3221), (3118, 3761), (3213, 3763)]:
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1
    g = geo.set_index("station_id").loc[ids]
    D = hav(g.lat.values[:, None], g.lon.values[:, None], g.lat.values[None], g.lon.values[None])
    A = np.clip(A + ((D <= 1500) & (D > 0)), 0, 1)
    return ids, A


def propagate(r, A, b, iters=2):
    deg = A.sum(1)
    out = r.copy()
    for _ in range(iters):
        nb = np.where(deg > 0, (A @ out) / np.maximum(deg, 1), out)
        out = (1 - b) * r + b * nb
    return out


def rmsle(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def recent_resid(B, anchor, feats):
    """anchor-56 까지로 학습해 (anchor-56, anchor] 를 예측한 역별 표본 외 잔차 (C&S 용)."""
    a2 = pd.Timestamp(anchor) - pd.Timedelta(days=56)
    tk = Tick("C&S 보조")
    tr2, _ = B.train_set(a2)
    va2 = B.rows(a2, pd.Timestamp(anchor))
    tk(f"피처 생성 train {len(tr2):,}")
    (p2,) = fit_predict(tr2, [va2], feats, [0], CONFIG["rounds"], B.cats)
    return va2.assign(e=p2 - va2.lb).groupby("station_id").e.mean()


def run_cv(train, meta, geo):
    B = Builder(train, meta, geo, CONFIG["groups"])
    feats = B.feats()
    ids, A = adjacency(meta, geo)
    summary = {}
    for name in CONFIG["folds"]:
        t0 = time.time()
        a, e = FOLDS[name]
        tk = Tick(f"fold {name}")
        tr, n = B.train_set(a)
        va = B.rows(a, pd.Timestamp(e))
        tk(f"피처 생성 anchors {n} train {len(tr):,} feats {len(feats)}")
        (pred,) = fit_predict(tr, [va], feats, CONFIG["seeds"], CONFIG["rounds"], B.cats)
        old = (va.is_new == 0).to_numpy()
        res = {"all": rmsle(pred, va.lb), "old": rmsle(pred[old], va.lb[old]),
               "new": rmsle(pred[~old], va.lb[~old]) if (~old).any() else None,
               "bias": float(np.mean(pred - va.lb))}
        err = pd.Series(pred - va.lb.to_numpy())
        res["by_daytype"] = {int(k): round(float(np.sqrt((v ** 2).mean())), 4) for k, v in err.groupby(va.daytype.to_numpy())}
        res["by_month"] = {int(k): round(float(np.sqrt((v ** 2).mean())), 4) for k, v in err.groupby(va.month.to_numpy())}
        out = va[["date", "station_id", "hour", "lb"]].assign(pred=pred)
        if CONFIG["cs"]:
            r = recent_resid(B, a, feats).reindex(ids).fillna(0).to_numpy()
            best = (res["old"], 0.0, 0.0)
            grid = {}
            for bb in (0.0, 0.3, 0.6, 0.9):
                rs = pd.Series(propagate(r, A, bb), index=ids)
                for aa in (0.25, 0.5, 0.75, 1.0):
                    adj = pred - aa * va.station_id.map(rs).to_numpy()
                    sc = rmsle(adj[old], va.lb[old])
                    grid[f"a{aa}_b{bb}"] = round(sc, 5)
                    if sc < best[0]:
                        best = (sc, aa, bb)
            res["cs_grid_old"] = grid
            res["cs_best_old"] = best
            out["resid_station"] = va.station_id.map(pd.Series(r, index=ids)).to_numpy()
        out.to_csv(OUT / f"cv_{CONFIG['tag']}_{name}.csv", index=False)
        res["n_anchor"], res["n_train"], res["sec"] = n, len(tr), round(time.time() - t0)
        summary[name] = res
        p(f"[{name}] all {res['all']:.5f} old {res['old']:.5f} new {res['new']} bias {res['bias']:+.4f} "
          f"| daytype {res['by_daytype']} | month {res['by_month']} | {res['sec']}s")
        if CONFIG["cs"]:
            p(f"    C&S best(old) {res['cs_best_old']}")
    json.dump(summary, open(OUT / f"cv_{CONFIG['tag']}.json", "w"), indent=1, ensure_ascii=False)


def run_full(train, test, meta, geo):
    B = Builder(train, meta, geo, CONFIG["groups"])
    feats = B.feats()
    t0 = time.time()
    last = train.date.max()
    tk = Tick("full")
    tr, n = B.train_set(last)
    te = B.rows(last, target=test)
    tk(f"피처 생성 anchors {n} train {len(tr):,} feats {len(feats)}")
    (pred,) = fit_predict(tr, [te], feats, CONFIG["seeds"], CONFIG["rounds"], B.cats)
    if CONFIG["cs"] and CONFIG.get("cs_a", 0) > 0:
        ids, A = adjacency(meta, geo)
        r = recent_resid(B, last, feats).reindex(ids).fillna(0).to_numpy()
        rs = pd.Series(propagate(r, A, CONFIG["cs_b"]), index=ids)
        pred = pred - CONFIG["cs_a"] * te.station_id.map(rs).to_numpy()
    sub = test[["example_id"]].merge(pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(np.clip(pred, 0, None))}),
                                     on="example_id", how="left")
    assert len(sub) == 157194 and sub.boardings.notna().all() and (sub.boardings >= 0).all()
    sub.to_csv(OUT / f"sub_{CONFIG['tag']}.csv", index=False)
    te[["example_id"]].assign(pred_log=pred).to_csv(OUT / f"testpred_{CONFIG['tag']}.csv", index=False)
    p(f"full: anchors {n} train {len(tr):,} feats {len(feats)} | {time.time() - t0:.0f}s")
    p(sub.boardings.describe().round(1).to_string())


def ev_file(kind):
    """경기 일정 파일: ev_actual 이면 실제 치러진 경기(취소 제외·추가 편성 포함), 아니면 시즌 전 공표 일정."""
    return f"{kind}_home_{'actual' if CONFIG.get('ev_actual') else 'schedule'}.csv"


_WX = None


def weather_feats(df):
    """인천 시간별 실측 날씨 (규칙 변경 11/11: 7/1 이후 메타데이터 허용, 정답 승차만 금지).
    wx_prcp: 해당 시간 강수, wx_prcp3: 직전 2시간 포함 3시간 합, wx_prcp_day: 그날 6~23시 합,
    wx_temp: 기온, wx_snow: 적설, wx_wind: 풍속."""
    global _WX
    if _WX is None:
        w = pd.read_csv(_find("weather_incheon_hourly.csv"), parse_dates=["date"]).sort_values(["date", "hour"])
        w["wx_prcp3"] = w.precipitation.rolling(3, min_periods=1).sum()
        day = w[w.hour.between(6, 23)].groupby("date").precipitation.sum().rename("wx_prcp_day")
        w = w.merge(day, on="date", how="left")
        _WX = w.rename(columns={"precipitation": "wx_prcp", "temperature_2m": "wx_temp", "snowfall": "wx_snow",
                                "wind_speed_10m": "wx_wind"}).set_index(["date", "hour"])[list(WX_COLS)]
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(df.date), df.hour.to_numpy()])
    out = _WX.reindex(idx)
    assert out.notna().all().all(), "날씨 자료 누락"
    return out.reset_index(drop=True).set_axis(df.index)


WX_COLS = ("wx_prcp", "wx_prcp3", "wx_prcp_day", "wx_temp", "wx_snow", "wx_wind")


def chronos_covariates(df, geo):
    """미래에 미리 알 수 있는 공변량 (달력·방학·사전 공표된 경기 일정)."""
    c = calendar_feats(df[["date", "station_id", "hour"]].assign(
        daytype=np.where(df.date.isin(HSET), 2, np.where(df.date.dt.dayofweek >= 5, 1, 0))))
    out = pd.DataFrame({
        "off": (c.daytype > 0).astype(float), "hol": (c.daytype == 2).astype(float),
        "bridge": c.bridge.astype(float), "after_off": c.after_long_off.astype(float),
        "bighol": np.where(c.big_hol_rel == 99, 0.0, 1.0 - np.abs(c.big_hol_rel) / 11.0),
        "univ_vac": c.univ_vac.astype(float), "school_vac": c.school_vac.astype(float)}, index=df.index)
    sch = pd.read_csv(_find(ev_file("ssg")), parse_dates=["date"]).set_index("date").start_hour
    st = df.date.map(sch).to_numpy(dtype=float)
    w = np.exp(-df.station_id.map(geo.set_index("station_id").dist_munhak).to_numpy() / 1000.0)
    rel = df.hour.to_numpy() - st
    out["ssg"] = np.where(~np.isnan(st) & (rel >= 0) & (rel <= 5), w, 0.0)
    if CONFIG.get("chr_bfc"):
        bs = pd.read_csv(_find(ev_file("bucheon")), parse_dates=["date"]).set_index("date").start_hour
        bst = df.date.map(bs).to_numpy(dtype=float)
        wb = np.exp(-df.station_id.map(geo.set_index("station_id").dist_bucheon_stadium).to_numpy() / 1000.0)
        rb = df.hour.to_numpy() - bst
        out["bfc"] = np.where(~np.isnan(bst) & (rb >= -1) & (rb <= 4), wb, 0.0)
    if CONFIG.get("chr_wx"):
        wx = weather_feats(df)
        out["prcp"] = np.log1p(wx.wx_prcp3.to_numpy())
        out["prcp_day"] = np.log1p(wx.wx_prcp_day.to_numpy())
        out["cold"] = np.clip(-wx.wx_temp.to_numpy(), 0, None) / 10.0
        out["hot"] = np.clip(wx.wx_temp.to_numpy() - 28, 0, None) / 5.0
        out["snow"] = np.log1p(wx.wx_snow.to_numpy() * 10)
    if CONFIG.get("chr_geo"):
        g = geo.set_index("station_id")
        lat, lon = df.station_id.map(g.lat).to_numpy(), df.station_id.map(g.lon).to_numpy()
        univ = [(37.4500, 126.6535), (37.3754, 126.6328), (37.3824, 126.6697)]  # 인하대, 인천대 송도, 연세대 국제캠
        prox = np.max([np.exp(-hav(lat, lon, a_, b_) / 1000.0) for a_, b_ in univ], axis=0)
        out["univ_vac_x"] = out.univ_vac.to_numpy() * prox
        out["univ_term_x"] = (1 - out.univ_vac.to_numpy()) * prox
        # 펜타포트 (사전 공표: 2025-08-01~03, 2026-07-31~08-02), 송도달빛축제공원 거리 감쇠
        fest = df.date.isin(pd.to_datetime(["2025-08-01", "2025-08-02", "2025-08-03",
                                            "2026-07-31", "2026-08-01", "2026-08-02"])).to_numpy()
        wf = np.exp(-df.station_id.map(g.dist_songdo_park).to_numpy() / 1000.0)
        out["pentaport"] = np.where(fest & (df.hour.to_numpy() >= 12), wf, 0.0)
    return out


def structural_ratio(train, geo, anchor, dates, K=10):
    """통행 목적 분해 기반 '달력이 설명하는 역·시간별 예상 변화율' (log).

    1) anchor 이전 기존역 [날짜 x (역,시간,승/하)] 인원수 행렬을 K개 목적으로 NMF(KL) 분해: X ≈ F·G
    2) 목적별 log 활동량 F[:,k] 를 달력 변수로 릿지 회귀 -> 과거·미래 모두 '달력 설명분' 만 사용 (성격 일치)
    3) 비율 = log((F̂[d]·G)[s,h]) - log((mean F · G)[s,h])   (평소 대비 예상 변화, 신규역은 0)
    반환: DataFrame(date, station_id, hour, struct)
    """
    from sklearn.decomposition import NMF
    from sklearn.linear_model import Ridge
    anchor = pd.Timestamp(anchor)
    new_st = train.groupby("station_id").date.min() > pd.Timestamp("2025-01-01")
    old_ids = sorted(new_st[~new_st].index)
    hist = train[(train.date <= anchor) & train.station_id.isin(old_ids)]
    cols_ = pd.MultiIndex.from_product([old_ids, range(6, 24)])
    B = hist.pivot_table(index="date", columns=["station_id", "hour"], values="boardings").reindex(columns=cols_)
    A = hist.pivot_table(index="date", columns=["station_id", "hour"], values="alightings").reindex(columns=cols_)
    X = np.hstack([B.to_numpy(), A.to_numpy()])
    ok = ~np.isnan(X).any(1)
    nmf = NMF(n_components=K, beta_loss="kullback-leibler", solver="mu", init="nndsvda", max_iter=600, random_state=0)
    Fd = nmf.fit_transform(X[ok])
    G = nmf.components_[:, :B.shape[1]]

    def design(ds):
        c = calendar_feats(pd.DataFrame({"date": ds, "daytype": np.where(pd.DatetimeIndex(ds).isin(HSET), 2,
                                                                           np.where(pd.DatetimeIndex(ds).dayofweek >= 5, 1, 0))}))
        D = pd.get_dummies(c.date.dt.dayofweek, prefix="dow").astype(float)
        for k_ in ("bridge", "after_long_off", "univ_vac", "school_vac"):
            D[k_] = c[k_].astype(float)
        D["hol"] = (c.daytype == 2).astype(float)
        D["bighol"] = np.where(c.big_hol_rel == 99, 0.0, 1.0 - np.abs(c.big_hol_rel) / 11.0)
        D["hol_x_vac"] = D.hol * D.school_vac
        return D.reindex(columns=[f"dow_{i}" for i in range(7)] + ["bridge", "after_long_off", "univ_vac",
                                                                     "school_vac", "hol", "bighol", "hol_x_vac"],
                         fill_value=0.0).to_numpy()
    reg = Ridge(alpha=1.0).fit(design(B.index[ok]), np.log(Fd + 1e-3))
    Fh = np.exp(reg.predict(design(pd.DatetimeIndex(dates))))
    base = np.log(Fd.mean(0) @ G + 1e-3)
    R = np.log(Fh @ G + 1e-3) - base  # (len(dates), S*H)
    out = pd.DataFrame(R, index=pd.DatetimeIndex(dates), columns=cols_).stack([0, 1], future_stack=True)
    out = out.rename("struct").reset_index()
    out.columns = ["date", "station_id", "hour", "struct"]
    return out


def chronos_predict(train, geo, anchor, target):
    """(역,시간) 일별 log 승차 시계열 1278개를 Chronos-2 로 예측. target: date/station_id/hour 프레임."""
    if CONFIG.get("chr_daily"):
        return chronos_daily_predict(train, geo, anchor, target)
    import torch
    from chronos import Chronos2Pipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global _PIPE
    if "_PIPE" not in globals():
        _PIPE = Chronos2Pipeline.from_pretrained(CONFIG.get("chronos_model", "amazon/chronos-2"), device_map=dev)
    pipe = _PIPE
    anchor = pd.Timestamp(anchor)
    ctx = train[train.date <= anchor][["date", "station_id", "hour", "lb", "la"]].copy()
    past = ["la"] if CONFIG.get("chr_la") else []  # 과거 전용 공변량 (미래 값 없음): 하차 log
    fut = target[["date", "station_id", "hour"]].drop_duplicates().copy()
    cols = None
    for df in (ctx, fut):
        cv = chronos_covariates(df, geo)
        cols = list(cv.columns)
        for c_ in cols:
            df[c_] = cv[c_].to_numpy()
        # item_id 사전순 = cross-learning 묶음 순서 (station 순: 노선상 이웃 / hour 순: 같은 시간대 전 역)
        hs = df.hour.map("{:02d}".format)
        df["item_id"] = (df.station_id.astype(str) + "_" + hs if CONFIG.get("cl_order", "station") == "station"
                         else hs + "_" + df.station_id.astype(str))
        df.rename(columns={"date": "timestamp"}, inplace=True)
    H = int((fut.timestamp.max() - anchor).days)
    fut = fut[fut.item_id.isin(set(ctx.item_id))]
    ctx = ctx[ctx.item_id.isin(set(fut.item_id))]  # Chronos 는 과거·미래의 시계열 ID 집합이 같아야 함
    if CONFIG.get("chr_struct"):
        t1 = time.time()
        alld = pd.DatetimeIndex(sorted(set(ctx.timestamp) | set(fut.timestamp)))
        st_ = structural_ratio(train, geo, anchor, alld, K=CONFIG["chr_struct"].get("K", 10))
        st_ = st_.rename(columns={"date": "timestamp"})
        ctx = ctx.merge(st_, on=["timestamp", "station_id", "hour"], how="left")
        fut = fut.merge(st_, on=["timestamp", "station_id", "hour"], how="left")
        ctx["struct"] = ctx.struct.fillna(0.0)
        fut["struct"] = fut.struct.fillna(0.0)
        cols = cols + ["struct"]
        p(f"  구조 공변량 K={CONFIG['chr_struct'].get('K', 10)} {time.time() - t1:.0f}s")
    ctx = ctx.sort_values(["item_id", "timestamp"], kind="stable")
    fut = fut.sort_values(["item_id", "timestamp"], kind="stable")
    ft = CONFIG.get("chr_ft")
    if ft:
        # anchor 이전 자료만으로 미세조정 (폴드마다 새로) -> 누출 없음
        from chronos.chronos2.preprocess import from_data_frame
        t0 = time.time()
        h_ft = ft.get("h", 63)
        inputs = from_data_frame(ctx[["item_id", "timestamp", "lb"] + cols], target_columns=["lb"],
                                 prediction_length=h_ft, known_covariates_names=cols)
        pipe = pipe.fit(inputs, prediction_length=h_ft, finetune_mode=ft.get("mode", "lora"),
                        learning_rate=ft.get("lr", 1e-4), num_steps=ft.get("steps", 1000),
                        batch_size=ft.get("batch", 256), min_past=ft.get("min_past", 63),
                        output_dir=str(OUT / "chr_ft_tmp"), remove_printer_callback=True)
        p(f"  chronos 미세조정 {ft} {time.time() - t0:.0f}s")
    t0 = time.time()
    pr = pipe.predict_df(ctx[["item_id", "timestamp", "lb"] + cols + past], future_df=fut[["item_id", "timestamp"] + cols],
                         target="lb", prediction_length=H, quantile_levels=[0.5],
                         cross_learning=CONFIG.get("cross_learning", False), batch_size=CONFIG.get("batch_size", 256),
                         context_length=CONFIG.get("context_length"))
    p(f"  chronos device={dev} series={ctx.item_id.nunique()} H={H} {time.time() - t0:.0f}s")
    pr = pr.rename(columns={"timestamp": "date"})
    sid_h = pr.item_id.str.split("_", expand=True).astype(int)
    if CONFIG.get("cl_order", "station") == "station":
        pr["station_id"], pr["hour"] = sid_h[0], sid_h[1]
    else:
        pr["station_id"], pr["hour"] = sid_h[1], sid_h[0]
    m = target[["date", "station_id", "hour"]].merge(pr[["date", "station_id", "hour", "predictions"]],
                                                     on=["date", "station_id", "hour"], how="left")
    return np.clip(m.predictions.to_numpy(), 0, None)


def chronos_daily_predict(train, geo, anchor, target):
    """역별 하루 총승차 log 시계열(71개)을 Chronos-2 로 예측 -> 최근 시간대 분포로 시간별 분배.
    시간대 분포: 평일·주말은 anchor 직전 8주, 공휴일은 전체 이력 공휴일 (없으면 주말)."""
    import torch
    from chronos import Chronos2Pipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global _PIPE
    if "_PIPE" not in globals():
        _PIPE = Chronos2Pipeline.from_pretrained(CONFIG.get("chronos_model", "amazon/chronos-2"), device_map=dev)
    anchor = pd.Timestamp(anchor)
    hist = train[train.date <= anchor][["date", "station_id", "hour", "boardings"]].copy()
    fut_h = target[["date", "station_id", "hour"]].drop_duplicates().copy()

    def daily(df_h, with_y):
        cv = chronos_covariates(df_h, geo)
        cv[["date", "station_id"]] = df_h[["date", "station_id"]].to_numpy()
        agg = cv.groupby(["date", "station_id"]).mean(numeric_only=True)
        if with_y:
            n = df_h.groupby(["date", "station_id"]).boardings.agg(["sum", "count"])
            agg["lb"] = np.log1p(n["sum"] * 18.0 / n["count"])  # 누락 시간 보정
        return agg.reset_index()
    ctx, fut = daily(hist, True), daily(fut_h, False)
    cols = [c for c in ctx.columns if c not in ("date", "station_id", "lb")]
    for d in (ctx, fut):
        d["item_id"] = d.station_id.astype(str)
        d.rename(columns={"date": "timestamp"}, inplace=True)
    fut = fut[fut.item_id.isin(set(ctx.item_id))]
    ctx = ctx[ctx.item_id.isin(set(fut.item_id))].sort_values(["item_id", "timestamp"])
    fut = fut.sort_values(["item_id", "timestamp"])
    H = int((fut.timestamp.max() - anchor).days)
    t0 = time.time()
    pr = _PIPE.predict_df(ctx[["item_id", "timestamp", "lb"] + cols], future_df=fut[["item_id", "timestamp"] + cols],
                          target="lb", prediction_length=H, quantile_levels=[0.5], batch_size=CONFIG.get("batch_size", 256))
    p(f"  chronos daily device={dev} series={ctx.item_id.nunique()} H={H} {time.time() - t0:.0f}s")
    pr = pr.rename(columns={"timestamp": "date"})
    pr["station_id"] = pr.item_id.astype(int)
    # 시간대 분포
    hist["t"] = np.where(hist.date.isin(HSET), 2, np.where(hist.date.dt.dayofweek >= 5, 1, 0))
    rec = hist[(hist.date > anchor - pd.Timedelta(days=56)) | (hist.t == 2)]
    tot = rec.groupby(["date", "station_id"]).boardings.transform("sum")
    rec = rec.assign(sh=rec.boardings / tot.clip(lower=1))
    S = rec.groupby(["station_id", "t", "hour"]).sh.mean().rename("sh").reset_index()
    S1 = S[S.t == 1].drop(columns="t").rename(columns={"sh": "sh1"})
    out = target[["date", "station_id", "hour"]].copy()
    out["t"] = np.where(out.date.isin(HSET), 2, np.where(out.date.dt.dayofweek >= 5, 1, 0))
    out = out.merge(pr[["date", "station_id", "predictions"]], on=["date", "station_id"], how="left")
    out = out.merge(S, on=["station_id", "t", "hour"], how="left").merge(S1, on=["station_id", "hour"], how="left")
    sh = out.sh.fillna(out.sh1).to_numpy()
    return np.log1p(np.expm1(out.predictions.to_numpy()) * sh)


def run_chronos(train, test, meta, geo):
    new_st = meta.set_index("station_id").first_date > "2025-01-01"
    if CONFIG["mode"] == "chronos_cv" and CONFIG.get("variants"):
        base_cfg = dict(CONFIG)
        for v in CONFIG["variants"]:
            CONFIG.clear()
            CONFIG.update(base_cfg, variants=None, **v)
            p(f"=== variant {v}")
            run_chronos(train, test, meta, geo)
        CONFIG.clear()
        CONFIG.update(base_cfg)
        return
    if CONFIG["mode"] == "chronos_cv":
        for name in CONFIG["folds"]:
            a, e = FOLDS[name]
            va = train[(train.date > a) & (train.date <= e)].reset_index(drop=True)
            pred = chronos_predict(train, geo, a, va)
            old = ~va.station_id.map(new_st).to_numpy()
            ok = ~np.isnan(pred)
            p(f"[{name}] chronos all {rmsle(pred[ok], va.lb[ok]):.5f} old {rmsle(pred[ok & old], va.lb[ok & old]):.5f} "
              f"missing {(~ok).sum()} bias {np.nanmean(pred - va.lb):+.4f}")
            va[["date", "station_id", "hour", "lb"]].assign(pred=pred).to_csv(OUT / f"cv_{CONFIG['tag']}_{name}.csv", index=False)
            if CONFIG.get("aux_resid"):
                # 보정용: anchor-56 기준으로 (anchor-56, anchor] 를 예측한 역별 표본 외 잔차
                a2 = pd.Timestamp(a) - pd.Timedelta(days=56)
                v2 = train[(train.date > a2) & (train.date <= a)].reset_index(drop=True)
                p2 = chronos_predict(train, geo, a2, v2)
                v2.assign(e=p2 - v2.lb).groupby("station_id").e.mean().rename("r").to_csv(
                    OUT / f"resid_{CONFIG['tag']}_{name}.csv")
    else:
        pred = chronos_predict(train, geo, train.date.max(), test)
        assert not np.isnan(pred).any(), "chronos test 예측 누락"
        if CONFIG.get("aux_resid"):
            a2 = train.date.max() - pd.Timedelta(days=56)
            v2 = train[train.date > a2].reset_index(drop=True)
            p2 = chronos_predict(train, geo, a2, v2)
            v2.assign(e=p2 - v2.lb).groupby("station_id").e.mean().rename("r").to_csv(OUT / f"resid_{CONFIG['tag']}.csv")
        test[["example_id"]].assign(pred_log=pred).to_csv(OUT / f"testpred_{CONFIG['tag']}.csv", index=False)
        p(f"chronos full: mean log {pred.mean():.4f}")


# ============================== 구조 모델 (통행 목적 x 왕복 x 평소 수준) ==============================
def struct_date_features(dates, geo=None):
    """날짜 단위 입력: 달력·방학·사전 공표 경기/축제 (목적별 활동량 신경망 입력)."""
    d = pd.DatetimeIndex(dates)
    c = calendar_feats(pd.DataFrame({"date": d, "daytype": np.where(d.isin(HSET), 2, np.where(d.dayofweek >= 5, 1, 0))}))
    X = pd.get_dummies(c.date.dt.dayofweek, prefix="dow").astype(float).reindex(
        columns=[f"dow_{i}" for i in range(7)], fill_value=0.0)
    X["hol"] = (c.daytype == 2).astype(float).to_numpy()
    for k in ("bridge", "after_long_off", "hol_prev", "hol_next", "univ_vac", "school_vac"):
        X[k] = c[k].astype(float).to_numpy()
    for r in range(-3, 4):
        X[f"bh{r}"] = (c.big_hol_rel == r).astype(float).to_numpy()
    X["term"] = np.nan_to_num(c.univ_term_day.to_numpy(dtype=float), nan=60.0) / 60.0
    sch = pd.read_csv(_find(ev_file("ssg")), parse_dates=["date"]).set_index("date").start_hour
    st = pd.Series(d).map(sch).to_numpy(dtype=float)
    X["ssg"] = (~np.isnan(st)).astype(float)
    for hh in (14, 17, 18):
        X[f"ssg{hh}"] = (st == hh).astype(float)
    bs = pd.read_csv(_find(ev_file("bucheon")), parse_dates=["date"]).set_index("date").start_hour
    bst = pd.Series(d).map(bs).to_numpy(dtype=float)
    X["bfc"] = (~np.isnan(bst)).astype(float)
    X["bfc_eve"] = (bst >= 19).astype(float)
    X["penta"] = d.isin(pd.to_datetime(["2025-08-01", "2025-08-02", "2025-08-03",
                                        "2026-07-31", "2026-08-01", "2026-08-02"])).astype(float)
    doy = d.dayofyear.to_numpy()
    for k in (1, 2):
        X[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return X.to_numpy(dtype=np.float32), c.daytype.to_numpy()


def struct_fit_predict(train, meta, geo, anchor, target_end):
    """anchor 이전 자료로 구조 모델 학습 -> (anchor, target_end] 승차 log1p 예측 [D_t, S, H].

    μ[d,s,h,c] = 평소수준_j[유형(d),s,h,c] x (Σ_k W[d,k]·L[k,s,h,c]) / (Σ_k W̄_j[유형(d),k]·L[k,s,h,c]) x exp(drift_s·horizon)
      W = softplus(MLP(날짜 입력))  : 목적별 활동량 (달력·방학·경기로 결정)
      L = softplus(param)           : 역·시간·승하차별 목적 비중 (역의 성격)
      W̄_j = 평소 수준 창(anchor j 직전) 의 평균 목적 활동량 -> 비율 = '평소 대비 구조적 변화'
      c ∈ {승차, 하차}: 같은 W 를 공유 -> 왕복 통행 구조로 하차 자료가 목적 식별을 돕는다
    손실: 포아송 (인원수 공간). anchor 이전 자료만 사용.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CONFIG.get("struct", {})
    Kp, steps, lr = cfg.get("K", 12), cfg.get("steps", 1500), cfg.get("lr", 0.02)
    anchor, target_end = pd.Timestamp(anchor), pd.Timestamp(target_end)
    d0 = train.date.min()
    dates = pd.date_range(d0, target_end)
    sids = np.array(sorted(meta.station_id))
    sidx = {x: i for i, x in enumerate(sids)}
    Dn, S, H = len(dates), len(sids), 18
    Y = np.full((Dn, S, H, 2), np.nan, np.float32)
    train = train[train.date <= anchor]  # anchor 이후 자료는 처음부터 넣지 않음 (누출 방지, 폴드 끝 이후 날짜 포함 방지)
    di = (train.date - d0).dt.days.to_numpy()
    si, hi = train.station_id.map(sidx).to_numpy(), train.hour.to_numpy() - 6
    Y[di, si, hi, 0] = train.boardings.to_numpy()
    Y[di, si, hi, 1] = train.alightings.to_numpy()
    Y[np.arange(Dn) > (anchor - d0).days] = np.nan  # anchor 이후 자료 차단 (누출 방지)
    a = (anchor - d0).days
    Xd, dt = struct_date_features(dates, geo)

    def baseline(ai):
        """anchor ai 시점 평소 수준 [3, S, H, 2] (평일/주말: 최근 8주, 공휴일: 전체 이력, 없으면 주말)."""
        Bm = np.full((3, S, H, 2), np.nan, np.float32)
        Wm = np.zeros((3, Dn), np.float32)
        for k in (0, 1, 2):
            lo = 0 if k == 2 else max(0, ai - 55)
            idx = np.arange(lo, ai + 1)
            idx = idx[dt[idx] == k]
            if len(idx):
                with np.errstate(all="ignore"):
                    Bm[k] = np.nanmean(Y[idx], axis=0)
                Wm[k, idx] = 1.0 / len(idx)
            all_idx = np.arange(0, ai + 1)
            all_idx = all_idx[dt[all_idx] == k]
            if len(all_idx):  # 최근 창에 자료 없는 역(신규역 등): 전체 이력 같은 유형 평균
                with np.errstate(all="ignore"):
                    Bm[k] = np.where(np.isnan(Bm[k]), np.nanmean(Y[all_idx], axis=0), Bm[k])
        Bm[2] = np.where(np.isnan(Bm[2]), Bm[1], Bm[2])
        if Wm[2].sum() == 0:
            Wm[2] = Wm[1]
        return Bm, Wm

    step = CONFIG.get("anchor_step", 7)
    anchors = list(range((pd.Timestamp("2025-02-26") - d0).days, a - 6, step))
    if CONFIG.get("smoke"):
        anchors = anchors[-4:]
    pairs, bases, wms = [], [], []
    for j, ai in enumerate(anchors):
        Bm, Wm = baseline(ai)
        bases.append(Bm)
        wms.append(Wm)
        pairs += [(j, dd) for dd in range(ai + 1, min(ai + HORIZON, a) + 1)]
    Bt, Wt = baseline(a)
    bases.append(Bt)
    wms.append(Wt)
    jt = len(anchors)
    te_days = np.arange(a + 1, Dn)

    def T(x, dtype=None):
        return torch.tensor(np.asarray(x), device=dev, dtype=dtype)
    Xd_t, dt_t = T(Xd), T(dt, torch.long)
    Bs = np.stack(bases)
    BOK = T(np.isfinite(Bs))                       # 평소 수준이 있는 칸 (개통 전 신규역 등은 False)
    BASE = T(np.where(np.isfinite(Bs), Bs, 1.0))   # 계산 경로에 NaN 이 없도록 채움 (역전파 0·NaN 방지)
    YOK = T(np.isfinite(Y))
    WM, Yt = T(np.stack(wms)), T(np.nan_to_num(Y))
    pj, pdd = T([q[0] for q in pairs], torch.long), T([q[1] for q in pairs], torch.long)
    hz = (pdd - T([anchors[q[0]] for q in pairs], torch.long)).float() / 365.0
    torch.manual_seed(cfg.get("seed", 0))
    net = torch.nn.Sequential(torch.nn.Linear(Xd.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, Kp)).to(dev)
    Lp = torch.nn.Parameter(torch.randn(Kp, S, H, 2, device=dev) * 0.1 - 2.0)
    drift = torch.nn.Parameter(torch.zeros(S, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [Lp, drift], lr=lr)
    sp = torch.nn.functional.softplus

    def mu(jj, dd, hzz):
        W = sp(net(Xd_t))                                   # [Dn, K]
        L = sp(Lp)                                          # [K, S, H, 2]
        num = torch.einsum("dk,kshc->dshc", W[dd], L)       # [B, S, H, 2]
        Wbar = torch.einsum("jtd,dk->jtk", WM, W)            # [J, 3, K]
        k_ = dt_t[dd]
        den = torch.einsum("bk,kshc->bshc", Wbar[jj, k_], L)
        return BASE[jj, k_] * num / den.clamp_min(1e-6) * torch.exp(drift[None, :, None, None] * hzz[:, None, None, None])

    t0 = time.time()
    nb = cfg.get("batch", 256)
    wa = torch.tensor([1.0, cfg.get("w_alight", 0.5)], device=dev)
    gen = torch.Generator().manual_seed(0)
    for it in range(steps):
        b = torch.randint(0, len(pairs), (nb,), generator=gen).to(dev)
        m_ = mu(pj[b], pdd[b], hz[b])
        y_ = Yt[pdd[b]]
        ok = YOK[pdd[b]] & BOK[pj[b], dt_t[pdd[b]]]
        m2 = m_.clamp_min(1e-3)
        nll = (m2 - y_ * torch.log(m2)) * wa
        loss = nll[ok].mean() + cfg.get("l2_drift", 1.0) * (drift ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"struct 학습 손실이 유한하지 않음 (step {it})")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % max(1, steps // 4) == 0:
            with torch.no_grad():
                e_ = (torch.log1p(m2[..., 0]) - torch.log1p(y_[..., 0]))[ok[..., 0]]
            p(f"    struct step {it + 1}/{steps} loss {loss.item():.3f} 학습 rmsle {e_.pow(2).mean().sqrt().item():.4f} "
              f"({time.time() - t0:.0f}s, {dev})")
    with torch.no_grad():
        dd = T(te_days, torch.long)
        jj = torch.full_like(dd, jt)
        out = mu(jj, dd, (dd - a).float() / 365.0)[..., 0]
        out = torch.where(BOK[jj, dt_t[dd]][..., 0], out, torch.full_like(out, float("nan")))
    return dates[te_days], sids, np.log1p(out.clamp_min(0).cpu().numpy())


def run_struct(train, test, meta, geo):
    def to_rows(dts, sids, P, target):
        D = pd.DataFrame(P.reshape(len(dts), -1), index=dts,
                         columns=pd.MultiIndex.from_product([sids, range(6, 24)])).stack([0, 1], future_stack=True)
        D = D.rename("pred").reset_index()
        D.columns = ["date", "station_id", "hour", "pred"]
        return target.merge(D, on=["date", "station_id", "hour"], how="left")
    new_st = meta.set_index("station_id").first_date > "2025-01-01"
    if CONFIG["mode"] == "struct_cv":
        for name in CONFIG["folds"]:
            a, e = FOLDS[name]
            dts, sids, P = struct_fit_predict(train, meta, geo, a, e)
            va = to_rows(dts, sids, P, train[(train.date > a) & (train.date <= e)][["date", "station_id", "hour", "lb"]])
            ok = va.pred.notna().to_numpy()
            old = ~va.station_id.map(new_st).to_numpy()
            if (~ok & old).any():
                raise RuntimeError(f"struct {name}: 기존역 예측 누락 {(~ok & old).sum()}행")
            p(f"[{name}] struct all {rmsle(va.pred[ok], va.lb[ok]):.5f} old {rmsle(va.pred[ok & old], va.lb[ok & old]):.5f} "
              f"missing {(~ok).sum()} bias {np.nanmean(va.pred - va.lb):+.4f}")
            va.to_csv(OUT / f"cv_{CONFIG['tag']}_{name}.csv", index=False)
    else:
        dts, sids, P = struct_fit_predict(train, meta, geo, train.date.max(), test.date.max())
        te = to_rows(dts, sids, P, test[["example_id", "date", "station_id", "hour"]])
        assert te.pred.notna().all(), "struct test 예측 누락"
        te[["example_id"]].assign(pred_log=te.pred.to_numpy()).to_csv(OUT / f"testpred_{CONFIG['tag']}.csv", index=False)
        p(f"struct full: mean log {te.pred.mean():.4f}")


if __name__ == "__main__":
    p("CONFIG", json.dumps(CONFIG, ensure_ascii=False))
    _t = time.time()
    tr_, te_, meta_, geo_ = load()
    p(f"load {time.time() - _t:.1f}s | cpu {os.cpu_count()} | gpu {gpu_available()}")
    if CONFIG["mode"].startswith("chronos"):
        run_chronos(tr_, te_, meta_, geo_)
    elif CONFIG["mode"].startswith("struct"):
        run_struct(tr_, te_, meta_, geo_)
    elif CONFIG["mode"] == "cv":
        run_cv(tr_, meta_, geo_)
    elif CONFIG["mode"] == "resid":
        # 최종 보정용: 6/30 기준 역별 표본 외 잔차만 계산해 저장
        B_ = Builder(tr_, meta_, geo_, CONFIG["groups"])
        r_ = recent_resid(B_, tr_.date.max(), B_.feats())
        r_.rename("r").to_csv(OUT / f"resid_{CONFIG['tag']}.csv")
        p(r_.describe().round(4).to_string())
    else:
        run_full(tr_, te_, meta_, geo_)
