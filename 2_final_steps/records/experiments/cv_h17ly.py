import numpy as np, pandas as pd, warnings, pickle; warnings.filterwarnings("ignore")
exec(open("cv_ly2.py",encoding="utf-8").read().split("b=r(P[o],y[o])")[0])
DATA=pickle.load(open("team_cv.pkl","rb")); t=DATA["B"]
dd=d[K].merge(t[K+["h13","h17"]],on=K,how="left")
for nm in ["h13","h17"]:
    Pm=dd[nm].to_numpy(); m=~np.isnan(Pm)&o; b=r(Pm[m],y[m])
    for ex,msk in [("excl ev st",~evst)]:
        ok=~np.isnan(A3)&m&~hol&msk; s=[f"{nm} base {b:.5f} | A3 {ex}:"]
        for w in (0.05,0.1,0.15,0.2): Q=np.where(ok,(1-w)*Pm+w*A3,Pm); s.append(f"w{w} {r(Q[m],y[m])-b:+.5f}")
        print(" ".join(s))
    # weekend level check: mean residual weekend vs weekday
    we=(d.date.dt.dayofweek>=5).to_numpy()&~hol&m
    print(f"   {nm} mean resid (y-P): weekday {np.mean(y[m&~we&~hol]-Pm[m&~we&~hol]):+.4f} weekend {np.mean(y[we]-Pm[we]):+.4f}")
