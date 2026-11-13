"""v10 experiment: two model families different from the existing blend, for folds A, B and the test period.
  lgbL : light LightGBM (organizer daily weather + calendar + 8-week level before the anchor date)
  mlp  : embedding MLP (train-only standardization, AdamW, LayerNorm, Dropout, cosine schedule,
         gradient clipping, early stopping on the last 8 weeks before the anchor, then refit on all history)
Target = log1p(boardings) - 8-week level (lvl); lvl is added back to the prediction.
Usage : python exp3_newmodels.py A|B|full [seed...]   (env MODELS=lgbL,mlp selects models)
Output: cv_<model>_<fold>.csv (date,station_id,hour,lb,pred) / testpred_<model>_full.csv (example_id,pred_log)
"""
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results_v10"
OUT.mkdir(exist_ok=True)
MODE = sys.argv[1]
SEEDS = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
K = ["date", "station_id", "hour"]
torch.set_num_threads(16)

tr = pd.read_csv(ROOT / "sincon-2026/train.csv", parse_dates=["date"])
te = pd.read_csv(ROOT / "sincon-2026/test.csv", parse_dates=["date"])
cal = pd.read_csv(ROOT / "calendar.csv", encoding="utf-8-sig", parse_dates=["date"]).drop(columns="holiday_name")
W = pd.concat([pd.read_csv(ROOT / f"weather_data/{f}_weather.csv", encoding="utf-8-sig", parse_dates=["date"])
               for f in ("train", "test")])
WX = ["avg_temp", "max_temp", "min_temp", "rainfall", "avg_wind", "sunshine", "humidity", "has_rain"]
W = W.groupby("date")[WX].mean().reset_index()
tr["lb"] = np.log1p(tr.boardings)
anchor = {"A": "2025-06-30", "B": "2026-02-28", "full": "2026-06-30"}[MODE]
end = {"A": "2025-10-31", "B": "2026-06-30", "full": None}[MODE]
anchor = pd.Timestamp(anchor)
hist = tr[tr.date <= anchor]
tgt = te[K + ["example_id"]].copy() if MODE == "full" else tr[(tr.date > anchor) & (tr.date <= pd.Timestamp(end))][K + ["lb"]].copy()


def build(anchor, hist, tgt):
    df = pd.concat([hist[K + ["lb"]].assign(is_tgt=0), tgt.assign(is_tgt=1)], ignore_index=True)
    df = df.merge(cal, on="date", how="left").merge(W, on="date", how="left")
    d = df.date
    df["dow"] = d.dt.day_of_week
    df["daytype"] = np.where(df.is_holiday == 1, 2, np.where(df.dow >= 5, 1, 0))
    df["off"] = (df.daytype > 0).astype(int)
    df["month"], df["doy"], df["dom"] = d.dt.month, d.dt.day_of_year, d.dt.day
    df["t_year"] = (d - pd.Timestamp("2025-01-01")).dt.days / 365.0
    rec = hist[hist.date > anchor - pd.Timedelta(days=56)].merge(cal[["date", "is_offday"]], on="date")
    L = rec.groupby(["station_id", "hour", "is_offday"]).lb.mean().rename("lvl").reset_index().rename(columns={"is_offday": "off"})
    Ld = rec.groupby(["station_id", "hour", rec.date.dt.day_of_week.rename("dow")]).lb.mean().rename("lvl_dow").reset_index()
    La = rec.groupby(["station_id", "hour"]).lb.mean().rename("lvl_all").reset_index()
    df = df.merge(L, on=["station_id", "hour", "off"], how="left").merge(Ld, on=["station_id", "hour", "dow"], how="left") \
           .merge(La, on=["station_id", "hour"], how="left")
    df["lvl"] = df.lvl.fillna(df.lvl_all)
    df["lvl_dow"] = df.lvl_dow.fillna(df.lvl)
    df["lvl"] = df.lvl.fillna(df.groupby("hour").lvl.transform("median"))
    df["lvl_dow"] = df.lvl_dow.fillna(df.lvl)
    df["lvl_dow_d"] = df.lvl_dow - df.lvl
    df["y"] = df.lb - df.lvl
    return df


df = build(anchor, hist, tgt)
CAL = ["is_holiday", "is_offday", "offday_run_len", "offday_day_idx", "offday_days_left", "is_before_long_offday",
       "is_after_long_offday", "is_univ_vacation", "is_school_vacation", "daytype", "dom", "doy", "t_year"]
NUM = CAL + WX + ["lvl", "lvl_dow_d"]
EMB = {"station_id": None, "hour": None, "dow": 7, "month": 13}
H, T = df[df.is_tgt == 0].copy(), df[df.is_tgt == 1].copy()
st_codes = {s: i for i, s in enumerate(sorted(df.station_id.unique()))}
for x in (H, T):
    x["st_i"] = x.station_id.map(st_codes)
    x["hr_i"] = x.hour - 6


def fit_lgb(seed):
    F = ["station_id", "hour", "dow", "month"] + NUM
    ds = lgb.Dataset(H[F], H.y, categorical_feature=["station_id"])
    m = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=127, min_data_in_leaf=50,
                       bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, verbose=-1, num_threads=16,
                       seed=seed), ds, num_boost_round=1500)
    return m.predict(T[F])


class Net(nn.Module):
    def __init__(self, n_num, n_st, hidden=256, drop=0.1):
        super().__init__()
        self.e_st, self.e_hr = nn.Embedding(n_st, 16), nn.Embedding(18, 8)
        self.e_dw, self.e_mo = nn.Embedding(7, 4), nn.Embedding(13, 4)
        self.e_sh = nn.Embedding(n_st * 18 * 3, 8)  # station x hour x day-type interaction
        d = n_num + 16 + 8 + 4 + 4 + 8
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(hidden, 1))

    def forward(self, xn, st, hr, dw, mo, dt):
        z = torch.cat([xn, self.e_st(st), self.e_hr(hr), self.e_dw(dw), self.e_mo(mo),
                       self.e_sh((st * 18 + hr) * 3 + dt)], 1)
        return self.body(z).squeeze(1)


def tens(x, mu, sd):
    return (torch.tensor(((x[NUM].to_numpy(float) - mu) / sd), dtype=torch.float32),
            torch.tensor(x.st_i.to_numpy()), torch.tensor(x.hr_i.to_numpy()), torch.tensor(x.dow.to_numpy()),
            torch.tensor(x.month.to_numpy()), torch.tensor(x.daytype.to_numpy()))


def train_mlp(Htr, seed, epochs, Hva=None, lr=2e-3):
    torch.manual_seed(seed)
    mu, sd = Htr[NUM].to_numpy(float).mean(0), Htr[NUM].to_numpy(float).std(0) + 1e-6  # train statistics only
    Xtr, ytr = tens(Htr, mu, sd), torch.tensor(Htr.y.to_numpy(), dtype=torch.float32)
    net = Net(len(NUM), len(st_codes))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    g = torch.Generator().manual_seed(seed + 1000)
    n, bs, hist_ = len(ytr), 2048, []
    best, best_ep = 1e9, epochs
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(n, generator=g).split(bs):
            opt.zero_grad(set_to_none=True)
            loss = ((net(*[a[idx] for a in Xtr]) - ytr[idx]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sch.step()
        if Hva is not None:
            net.eval()
            with torch.no_grad():
                p = net(*tens(Hva, mu, sd)).numpy()
            v = float(np.sqrt(np.mean((p - Hva.y.to_numpy()) ** 2)))
            hist_.append(v)
            if v < best:
                best, best_ep = v, ep + 1
    return net, mu, sd, best_ep, hist_


def fit_mlp(seed):
    # 1) hold out the last 8 weeks before the anchor to pick the epoch count (time split, no leakage)
    cut = anchor - pd.Timedelta(days=56)
    _, _, _, best_ep, hv = train_mlp(H[H.date <= cut], seed, 16, H[H.date > cut])
    # 2) refit on the full history for that many epochs
    net, mu, sd, _, _ = train_mlp(H, seed, best_ep)
    net.eval()
    with torch.no_grad():
        p = net(*tens(T, mu, sd)).numpy()
    print(f"  mlp seed{seed}: inner-val {[round(x, 4) for x in hv]} -> epoch {best_ep}", flush=True)
    return p


MODELS = os.environ.get("MODELS", "lgbL,mlp").split(",")
for name, fn in [(n, f) for n, f in (("lgbL", fit_lgb), ("mlp", fit_mlp)) if n in MODELS]:
    t0 = time.time()
    ps = [fn(s) for s in SEEDS]
    pred = np.mean(ps, axis=0) + T.lvl.to_numpy()
    if MODE == "full":
        T[["example_id"]].assign(pred_log=pred).to_csv(OUT / f"testpred_{name}_full.csv", index=False)
        print(f"{name} full saved, mean {pred.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    else:
        r = lambda p: np.sqrt(np.mean((p - T.lb.to_numpy()) ** 2))
        each = [round(r(p + T.lvl.to_numpy()), 4) for p in ps]
        print(f"{name} fold {MODE}: per-seed {each} seed-avg {r(pred):.4f} ({time.time() - t0:.0f}s)", flush=True)
        T[K + ["lb"]].assign(pred=pred).to_csv(OUT / f"cv_{name}_{MODE}.csv", index=False)
