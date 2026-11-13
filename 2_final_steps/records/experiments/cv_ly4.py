import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("cv_ly2.py",encoding="utf-8").read().split("b=r(P[o],y[o])")[0])
ok=~np.isnan(A3)&o&~hol&~evst; b=r(P[o],y[o]); mo=d.date.dt.month.to_numpy()
for m in (3,4,5,6):
    s=[f"month {m}:"]
    for w in (0,0.1,0.2,0.3,0.4):
        mm=ok&(mo==m); Q=np.where(mm,(1-w)*P+w*A3,P); s.append(f"w{w} {r(Q[o&(mo==m)],y[o&(mo==m)]):.5f}")
    print(" ".join(s))
