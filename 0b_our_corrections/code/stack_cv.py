"""v8d 혼합 위에 상황별 보정(스태킹) — 폴드 하나 빼고 학습(LOFO) 검증.
타깃: lb - P_v8d. 입력: 각 모델 예측과 P 의 차이, 보정항, 요일유형·시간·예측거리.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "../code")
os.environ["SINCON_OUT"] = "../output/tmp"
sys.argv = sys.argv[:1]
import sincon_v2 as V  # noqa: E402

K = ["date", "station_id", "hour"]
NEW = [3107, 3108, 3109]
TAGS = ["lt1", "lt2", "chr_e3", "wx_chr", "x10", "x8"]
W8 = dict(wl=0.75, we3=0.5, wt=0.5, kw=1.0, kg=0.4, mh=0.5)
FE = ["d_lgb", "d_xgb", "d_e3", "d_wx", "spread", "wx", "game", "dt", "hour", "dow", "hz", "hol_near", "lvl"]


def r(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def v8d_blend(d):
    lgbm = (d.lt1 + d.lt2) / 2
    xgb = (d.x10 + d.x8) / 2
    tree = W8["wl"] * lgbm + (1 - W8["wl"]) * xgb
    chron = W8["we3"] * d.chr_e3 + (1 - W8["we3"]) * d.wx_chr
    P = W8["wt"] * tree + (1 - W8["wt"]) * chron + W8["kw"] * d.wx + W8["kg"] * d.game
    h = d.htg.to_numpy()
    P = np.where(np.isnan(h), P, (1 - W8["mh"]) * P + W8["mh"] * np.nan_to_num(h))
    return np.clip(P, 0, None), lgbm, xgb


def feats(d, anchor):
    P, lgbm, xgb = v8d_blend(d)
    X = pd.DataFrame({"P": P})
    X["d_lgb"] = lgbm - P
    X["d_xgb"] = xgb - P
    X["d_e3"] = d.chr_e3 - P
    X["d_wx"] = d.wx_chr - P
    X["spread"] = np.std(np.stack([lgbm, xgb, d.chr_e3, d.wx_chr]), axis=0)
    X["wx"] = d.wx.to_numpy()
    X["game"] = d.game.to_numpy()
    X["dt"] = np.where(d.date.isin(V.HSET), 2, np.where(d.date.dt.dayofweek >= 5, 1, 0))
    X["hour"] = d.hour.to_numpy()
    X["dow"] = d.date.dt.dayofweek.to_numpy()
    X["hz"] = (d.date - pd.Timestamp(anchor)).dt.days.to_numpy()
    X["hol_near"] = (~np.isnan(d.htg.to_numpy())).astype(int)
    X["lvl"] = P
    return X


def load_fold(f, src="out"):
    d = pd.read_parquet(f"out/post_{f}.parquet")
    for t in TAGS:
        x = pd.read_csv(f"{src}/cv_{t}_{f}.csv", parse_dates=["date"])[K + ["pred"]].rename(columns={"pred": t})
        d = d.merge(x, on=K, how="left")
    return d


PRM = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.8,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)

if __name__ == "__main__":
    data = {}
    for f in "ABC":
        d = load_fold(f)
        X = feats(d, V.FOLDS[f][0])
        X["y"] = d.lb.to_numpy() - X.P.to_numpy()
        X["old"] = ~d.station_id.isin(NEW).to_numpy()
        X["lb"] = d.lb.to_numpy()
        data[f] = X
    for rounds in (100, 200, 400):
        for shrink in (0.5, 1.0):
            out = []
            for f in "ABC":
                trn = pd.concat([data[g] for g in "ABC" if g != f])
                trn = trn[trn.old]
                m = lgb.train(PRM, lgb.Dataset(trn[FE], trn.y), rounds)
                te = data[f]
                p = np.clip(te.P + shrink * m.predict(te[FE]), 0, None)
                o = te.old.to_numpy()
                out.append((r(te.P[o], te.lb[o]), r(p[o], te.lb[o])))
            print(f"rounds {rounds} shrink {shrink}: " + " | ".join(
                f"{f} {a:.5f}->{b:.5f} ({b - a:+.5f})" for f, (a, b) in zip("ABC", out)), flush=True)
