"""v8d(또는 같은 파이프라인의 제출 파일)에 재학습 없이 얹는 추가 보정.

  A. 명절 보정 오염 제거 : 2025 준PO(10/9) 급증이 추석 패턴에 섞여 9/28(추석+3) 문학 주변 역을 부풀린 것 되돌림
  B. 부천 경기 보정 재조정: 2026(K리그1) 경기만으로 잰 실제 경기일 효과와, 제출 파일에 이미 들어간 효과의 차이를 절반 반영
  C. 월별 수준 보정        : 일반 평일(공휴일 제외)의 전년 대비 비율을 상반기 평균(약 +3%)으로 alpha 만큼 당김

사용: python post_v9.py <train.csv> <test.csv> <aux_data 폴더> <sincon_v2.py> <입력 제출.csv> <출력 폴더>
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN, TEST, AUX, SRC, SUB, OUT = sys.argv[1:7]
AUX, OUT = Path(AUX), Path(OUT)
OUT.mkdir(parents=True, exist_ok=True)

src = open(SRC, encoding="utf-8").read()
HOL = pd.to_datetime(re.findall(r'"(20\d\d-\d\d-\d\d)"', src.split("HOLIDAYS = pd.to_datetime([")[1].split("])")[0]))
BIG = pd.to_datetime(re.findall(r'"(20\d\d-\d\d-\d\d)"', src.split("BIG_HOLIDAYS = pd.to_datetime([")[1].split("])")[0]))
HSET = set(HOL)

tr = pd.read_csv(TRAIN, parse_dates=["date"])
te = pd.read_csv(TEST, parse_dates=["date"])
sub = pd.read_csv(SUB)
assert (sub.example_id.values == te.example_id.values).all()
geo = pd.read_csv(AUX / "station_geo_graph.csv").set_index("station_id")
tr["lb"] = np.log1p(tr.boardings)
P0 = np.log1p(sub.boardings.to_numpy())
ANCHOR = tr.date.max()
K = ["date", "station_id", "hour"]


def typ(d):
    return (d.date.isin(HSET) | (d.date.dt.dayofweek >= 5)).astype(int)


# ---------------- A. 명절 보정 오염 제거 ----------------
def holiday_target(h_src):
    """final_post.holiday_target 과 같은 계산 (h_src 의 행만 사용)."""
    def lvl(h, end):
        x = h[(h.date > end - pd.Timedelta(days=28)) & (h.date <= end)]
        return x.groupby(["station_id", "hour", "t"]).lb.median().rename("lvl")
    h = h_src[["date", "station_id", "hour", "lb"]].copy()
    h["t"] = typ(h)
    devs = []
    for b in BIG:
        if b + pd.Timedelta(days=3) > ANCHOR:
            continue
        x = h[(h.date - b).dt.days.abs() <= 3].copy()
        x["rel"] = (x.date - b).dt.days
        x = x.merge(lvl(h, b - pd.Timedelta(days=11)).reset_index(), on=["station_id", "hour", "t"], how="left")
        x["dev"] = x.lb - x.lvl
        devs.append(x[["station_id", "hour", "rel", "dev"]])
    D = pd.concat(devs).groupby(["station_id", "hour", "rel"]).dev.mean().rename("hdev").reset_index()
    d = te[K].copy()
    d["t"] = typ(d)
    rel = pd.Series(99, index=d.index)
    for b in BIG:
        r = (d.date - b).dt.days
        rel = rel.where(~((r.abs() <= 3) & (rel == 99)), r)
    d["rel"] = rel
    d = d.merge(D, on=["station_id", "hour", "rel"], how="left").merge(
        lvl(h, ANCHOR).reset_index(), on=["station_id", "hour", "t"], how="left")
    tg = (d.lvl + d.hdev).to_numpy()
    return np.where((d.rel != 99).to_numpy() & ~np.isnan(tg), tg, np.nan)


POSTSEASON = pd.to_datetime(["2025-10-09", "2025-10-10", "2025-10-11"])  # 준PO 1·2차전 + 우천 순연일
near_munhak = geo.index[geo.dist_munhak < 3000].tolist()
clean = tr[~(tr.date.isin(POSTSEASON) & tr.station_id.isin(near_munhak))]
h_old, h_new = holiday_target(tr), holiday_target(clean)
# 원래 파이프라인: P = 0.5*P + 0.5*htg  ->  htg 만 바뀌면 P 차이 = 0.5*(new-old)
dA = np.where(np.isnan(h_old) | np.isnan(h_new), 0.0, 0.5 * (h_new - h_old))
print(f"[A] 문학 3km 역 {near_munhak} | 바뀐 행 {(np.abs(dA) > 1e-9).sum()} | 최대 감소 {dA.min():+.3f}")


# ---------------- B. 부천 경기 보정 재조정 ----------------
bfc = pd.read_csv(AUX / "bucheon_home_actual.csv", parse_dates=["date"]).set_index("date").start_hour
near_bfc = geo.index[geo.dist_bucheon_stadium < 3000].tolist()


def game_dev(df, val):
    """경기일 값 - 같은 역·시간·요일유형의 ±10일 비경기일 중앙값."""
    d = df[df.station_id.isin(near_bfc)][K + [val]].copy()
    d["t"] = typ(d)
    d["st"] = d.date.map(bfc)
    d["rel"] = d.hour - d.st
    out = []
    for g in d.loc[d.st.notna(), "date"].unique():
        w = d[(d.date - g).abs().dt.days.le(10) & d.st.isna()]
        tg = d[d.date == g]
        base = w[w.t == tg.t.iloc[0]].groupby(["station_id", "hour"])[val].median().rename("base")
        out.append(tg.merge(base.reset_index(), on=["station_id", "hour"]))
    x = pd.concat(out)
    x["dev"] = x[val] - x.base
    return x


hist = game_dev(tr[tr.date >= "2026-01-01"], "lb")       # 2026 K리그1 실제 효과
H = hist.groupby(["station_id", "t", "rel"]).dev.agg(["sum", "count"])
H = (H["sum"] / (H["count"] + 2.0)).rename("h_eff").reset_index()
pred = game_dev(te.assign(p=P0), "p")                      # 제출 파일에 들어간 효과
pred = pred.merge(H, on=["station_id", "t", "rel"], how="left")
pred["gap"] = (pred.h_eff - pred.dev).where(pred.rel.between(-1, 5), 0.0).fillna(0.0)
print("[B] 2026 실제 효과 vs 제출 파일 효과 (3754, rel별 평균):")
print(pred[pred.station_id == 3754].groupby("rel")[["h_eff", "dev", "gap"]].mean().round(2).T.to_string())
gapB = te[K].merge(pred[K + ["gap"]], on=K, how="left").gap.fillna(0.0).to_numpy()


# ---------------- C. 월별 수준 (일반 평일) ----------------
OLD = ~te.station_id.isin([3107, 3108, 3109])
wd_te = (te.date.dt.dayofweek < 5) & ~te.date.isin(HSET)


def wd_level(df, val):
    d = df[~df.station_id.isin([3107, 3108, 3109]) & (df.date.dt.dayofweek < 5) & ~df.date.isin(HSET)]
    g = d.groupby([d.date.dt.year, d.date.dt.month])
    return g[val].sum() / g.date.nunique()


lv = wd_level(tr, "boardings")
target = np.mean([lv[(2026, m)] / lv[(2025, m)] for m in range(1, 7)])
lp = wd_level(te.assign(b=np.expm1(P0)), "b")
ratio = {m: lp[(2026, m)] / lv[(2025, m)] for m in range(7, 11)}
print(f"[C] 상반기 평일 전년비 {target:.4f} | 예측 평일 전년비 " + ", ".join(f"{m}월 {r:.4f}" for m, r in ratio.items()))
logf = te.date.dt.month.map({m: np.log(target / r) for m, r in ratio.items()}).to_numpy()
maskC = (OLD & wd_te).to_numpy()


def save(name, P):
    s = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(np.clip(P, 0, None))})
    assert len(s) == len(te) and s.boardings.notna().all()
    s.to_csv(OUT / name, index=False)
    d = P - P0
    print(f"  {name:28s} 바뀐 행 {(np.abs(d) > 1e-9).sum():6d} | 로그 변화 평균 {d.mean():+.4f} RMS {np.sqrt((d**2).mean()):.4f}")


print("\n출력:")
# B 는 진단용: v8d 가 이미 2026 경기 효과를 거의 그대로 반영 (정점 1.91 vs 실제 2.05) 하므로 제출에 쓰지 않음
save("v9_A.csv", P0 + dA)
save("v9_A_C50.csv", P0 + dA + np.where(maskC, 0.5 * logf, 0.0))
save("v9_A_C100.csv", P0 + dA + np.where(maskC, logf, 0.0))
