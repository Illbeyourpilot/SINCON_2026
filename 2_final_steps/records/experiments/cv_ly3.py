import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("cv_ly2.py",encoding="utf-8").read().split("b=r(P[o],y[o])")[0])
ok=~np.isnan(A3)&o&~hol&~evst; b=r(P[o],y[o])
t=np.where(d.date.dt.dayofweek>=5,"we","wd"); hb=pd.cut(d.hour,[5,9,16,20,23],labels=["06-09","10-16","17-20","21-23"]).astype(str).to_numpy()
for grp,lab in [("type",t),("hour",hb)]:
    for g in sorted(set(lab)):
        s=[f"{grp} {g}:"]
        for w in (0.1,0.2,0.3,0.4):
            m=ok&(lab==g); Q=np.where(m,(1-w)*P+w*A3,P); s.append(f"w{w} {r(Q[o],y[o])-b:+.5f}")
        print(" ".join(s))
