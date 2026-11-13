import numpy as np, pandas as pd, warnings, os; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
C=["q_align","q_base","e_norm","e_nbr","cb","lt1s2","lt2s3"]
res={}
for f in "ABC":
    d=load_fold(f); P,lg,xg=v8d_blend(d); y=d.lb.to_numpy(); o=~d.station_id.isin([3107,3108,3109]).to_numpy()
    tree=0.75*lg+0.25*xg; b=r(P[o],y[o])
    for t in C:
        fn=f"out/cv_{t}_{f}.csv"
        if not os.path.exists(fn): continue
        x=d[K].merge(pd.read_csv(fn,parse_dates=["date"])[K+["pred"]],on=K,how="left").pred.to_numpy()
        ok=~np.isnan(x)
        res[(t,"alone",f)]=r(x[o&ok],y[o&ok])
        for w in (0.2,0.35,0.5):
            nt=np.where(ok,(1-w)*tree+w*x,tree); Q=P+0.5*(nt-tree)
            res[(t,w,f)]=r(Q[o],y[o])-b
for t in C:
    fs=[f for f in "ABC" if (t,0.2,f) in res]
    if not fs: continue
    print(f"{t:8s} alone "+" ".join(f"{f} {res[(t,'alone',f)]:.4f}" for f in fs))
    for w in (0.2,0.35,0.5): print(f"   w{w}: "+" | ".join(f"{f} {res[(t,w,f)]:+.5f}" for f in fs))
