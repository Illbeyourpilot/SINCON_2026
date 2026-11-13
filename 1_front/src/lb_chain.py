"""v8dh17 류 raw 제출 -> 팀원 블렌드 + 보정 + 연습대회 수준 조정(조정 1~5) 적용.
사용: python lb_chain.py <raw.csv> <out.csv>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); _a = sys.argv[1:]; sys.argv = ["x"]
import sincon_v2 as V  # noqa: E402
TM = ROOT / "teammate"
te = pd.read_csv(ROOT / "data/test.csv", parse_dates=["date"]); tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
L = lambda f: np.log1p(te[["example_id"]].merge(pd.read_csv(f), on="example_id").boardings.to_numpy())
NEW = [3107, 3108, 3109]
wdm = lambda df: ~df.station_id.isin(NEW) & (df.date.dt.dayofweek < 5) & ~df.date.isin(V.HSET)
act = tr[wdm(tr)].groupby("date").boardings.sum()
target = act["2026-05-01":"2026-06-30"].mean() / act["2025-05-01":"2025-06-30"].mean()


def weekC(p, win=7):
    t = te.assign(b=np.expm1(p)); pred = t[wdm(t)].groupby("date").b.sum(); adj = {}
    for day in pred.index:
        pw = pred[day - pd.Timedelta(days=win):day + pd.Timedelta(days=win)]
        lw = act.reindex([x - pd.Timedelta(days=364) for x in pw.index]); ok = lw.notna().to_numpy()
        adj[day] = np.log(target * lw[ok].mean() / pw[ok].mean())
    m = ((te.date <= "2026-08-31") & wdm(te)).to_numpy()
    return np.where(m, te.date.map(adj).fillna(0).to_numpy(), 0)


import os
C = {k: float(os.environ.get("LC_" + k.upper(), v)) for k, v in dict(k=0.65, g=-0.01, jul=0.03, sep=0.02, ln=0.03, mid=0.01).items()}


def chain(new, k_other=None):
    k_other = C["k"] if k_other is None else k_other
    old = L(ROOT / "out/sub_v8dh13.csv")
    blend = L(TM / "input/submission_blend_base60_v10_40.csv"); team = L(TM / "output/blend60_ourfix_safe.csv")
    other = (team - blend) - weekC(blend)
    b = blend + 0.6 * (new - old) + k_other * other
    x = b + weekC(b)
    mo, h = te.date.dt.month.to_numpy(), te.hour.to_numpy()
    x = x + C["g"] + np.where(mo == 7, C["jul"], 0) - np.where(mo == 9, C["sep"], 0)
    x = x - np.where(h >= 22, C["ln"], 0) + np.where((h >= 10) & (h <= 16), C["mid"], 0)
    return x


if __name__ == "__main__":
    x = chain(L(_a[0]))
    s = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(np.clip(x, 0, None))})
    assert len(s) == 157194 and s.boardings.notna().all(); s.to_csv(_a[1], index=False); print("저장", _a[1])
