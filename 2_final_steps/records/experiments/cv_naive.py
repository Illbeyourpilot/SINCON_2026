import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
tr,_,_,_=V.load(); tr["lb"]=np.log1p(tr.boardings); HS=V.HSET
def dtyp(s): return np.where(s.dt.dayofweek==5,1,np.where(s.dt.dayofweek==6,2,0))
res={}
for f in "ABC":
    d=load_fold(f); P,_,_=v8d_blend(d); y=d.lb.to_numpy(); a=pd.Timestamp(V.FOLDS[f][0])
    o=~d.station_id.isin([3107,3108,3109]).to_numpy()
    for W in (28,56):
        h=tr[(tr.date<=a)&(tr.date>a-pd.Timedelta(days=W))&~tr.date.isin(HS)].copy(); h["t"]=dtyp(h.date)
        nv=h.groupby(["station_id","t","hour"]).lb.median()
        key=pd.MultiIndex.from_arrays([d.station_id,dtyp(d.date),d.hour]); N=nv.reindex(key).to_numpy()
        m=o&~d.date.isin(HS).to_numpy()&~np.isnan(N)
        for w in (0,0.05,0.1,0.15,0.2,0.3):
            Q=P.copy(); Q[m]=(1-w)*P[m]+w*N[m]
            res[(f,W,w)]=r(Q[o],y[o])
        res[(f,W,"naive")]=r(N[m],y[m])
for W in (28,56):
    print(f"W{W} naive alone:"," ".join(f"{f} {res[(f,W,'naive')]:.4f}" for f in "ABC"))
    for w in (0,0.05,0.1,0.15,0.2,0.3):
        print(f"  w{w}: "+" | ".join(f"{f} {res[(f,W,w)]:.5f} ({res[(f,W,w)]-res[(f,W,0)]:+.5f})" for f in "ABC"))
