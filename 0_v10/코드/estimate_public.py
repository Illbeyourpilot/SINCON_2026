"""Estimate public-LB RMSLE of candidate submissions without submitting.

Idea (all in log1p space, e8 = v8d prediction error on the test rows, unknown):
    LB(c)^2 = LB(v8d)^2 + |d_c|^2/n + 2<e8, d_c>/n,   d_c = c - v8d
- <e8, d_h9> is known exactly from the two scored files (v8d 0.15475, h9 0.153022).
- <e8, Delta> for Delta = v10 - v8d is estimated from validation folds A/B:
    rho = <e8_val, Delta_val> / |Delta_val|^2  ->  <e8, Delta_test> ~= rho * |Delta_test|^2
Assumes the public split behaves like the whole test period (unknown public fraction).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
V9 = ROOT / "SINCON_v9_실험"
K = ["date", "station_id", "hour"]
LB = {"v8d": 0.15475, "h9": 0.153022}


def lg(path):
    return np.log1p(pd.read_csv(path).boardings.to_numpy())


# ---- validation: rho per fold ----
D = pd.read_pickle(V9 / "cv_all.pkl")
rho, gain = {"v10": {}, "v9b": {}, "v11": {}}, {}
for fold in "AB":
    d = D[fold].merge(pd.read_csv(V9 / f"results_v10/cv_lgbL_{fold}.csv")[K + ["pred"]], on=K)
    v8 = 0.5 * (0.75 * (d.lt1 + d.lt2) / 2 + 0.25 * (d.v3_x10 + d.v3_x8) / 2) + 0.25 * (d.chr_e3 + d.wx_chr)
    v9b = 0.5 * (d.lt1 + d.lt2) / 2 + 0.25 * (d.chr_e3 + d.wx_chr_st)
    e = (v8 - d.lb).to_numpy()
    lagp = pd.read_csv(V9 / f"results_v11/cv_lag_{fold}.csv")[K + ["pred"]].rename(columns={"pred": "lagp"})
    lagp = d[K].merge(lagp, on=K, how="left").lagp.to_numpy()
    for key, c in (("v10", 0.8 * v9b + 0.2 * d.pred), ("v9b", v9b), ("v11", 0.7 * v9b + 0.3 * lagp)):
        dl = (c - v8).to_numpy()
        rho[key][fold] = float(e @ dl / (dl @ dl))
        gain[(key, fold)] = (np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean((e + dl) ** 2)))

# ---- test-period files ----
F = ROOT / "최종_v10"
v8 = lg(F / "입력/submission_v8d.csv")
h9 = lg(F / "입력/sub_v8dh9.csv")
cands = {"v10": lg(F / "제출파일/submission_v10.csv"), "h9_v10": lg(F / "제출파일/submission_h9_v10.csv")}
v9b_path = ROOT / "이전_제출후보/submission_v9b.csv"
v11_dir = ROOT / "최종_v11" / "제출파일"
if v11_dir.exists():
    cands["v11"] = lg(v11_dir / "submission_v11.csv")
    cands["h9_v11"] = lg(v11_dir / "submission_h9_v11.csv")
if v9b_path.exists():
    cands["v9b"] = lg(v9b_path)
    cands["h9_v9b"] = lg(ROOT / "이전_제출후보/submission_h9_v9b.csv")
n = len(v8)
S8 = LB["v8d"] ** 2
dh = h9 - v8
e_dh = (LB["h9"] ** 2 - S8 - dh @ dh / n) * n / 2  # exact <e8, d_h9>

for key in rho:
    print(f"validation {key}: rho_A={rho[key]['A']:.3f} rho_B={rho[key]['B']:.3f}  "
          + "  ".join(f"{f}: {gain[(key, f)][0]:.4f}->{gain[(key, f)][1]:.4f}" for f in "AB"))
print(f"anchors: v8d {LB['v8d']:.5f}, h9 {LB['h9']:.6f}")
rows = []
for name, c in cands.items():
    dc = c - v8
    # split d_c into the h9 part (exactly known) and the rest (estimated with rho)
    share_h9 = name.startswith("h9_")
    rest = dc - dh if share_h9 else dc
    known = e_dh if share_h9 else 0.0
    rr = rho[name.replace("h9_", "")]
    est = {}
    for tag, r in (("rhoA", rr["A"]), ("rhoB", rr["B"]), ("mean", (rr["A"] + rr["B"]) / 2)):
        s2 = S8 + dc @ dc / n + 2 * (known + r * (rest @ rest)) / n
        est[tag] = np.sqrt(max(s2, 0))
    rows.append((name, est["mean"], min(est["rhoA"], est["rhoB"]), max(est["rhoA"], est["rhoB"])))
out = pd.DataFrame(rows, columns=["candidate", "est_public", "low", "high"]).sort_values("est_public")
print(out.round(5).to_string(index=False))
