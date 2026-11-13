import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
tr,_,_,_=V.load(); tr["lb"]=np.log1p(tr.boardings); HS=V.HSET
f="B"; d=load_fold(f); P,_,_=v8d_blend(d); y=d.lb.to_numpy(); a=pd.Timestamp(V.FOLDS[f][0])
o=~d.station_id.isin([3107,3108,3109]).to_numpy()
L=tr.set_index(["station_id","date","hour"]).lb
def wd(df): return df[(df.date.dt.dayofweek<5)&~df.date.isin(HS)]
def lv(df,lo,hi): x=df[df.date.between(lo,hi)]; return x.groupby("station_id").boardings.sum()/x.date.nunique()
for tw in [("2026-01-01","2026-02-28","2025-01-01","2025-02-28")]:
    g=np.log(lv(wd(tr),tw[0],tw[1])/lv(wd(tr),tw[2],tw[3]))
    gl=float(np.log(wd(tr)[wd(tr).date.between(tw[0],tw[1])].boardings.sum()/wd(tr)[wd(tr).date.between(tw[0],tw[1])].date.nunique()/(wd(tr)[wd(tr).date.between(tw[2],tw[3])].boardings.sum()/wd(tr)[wd(tr).date.between(tw[2],tw[3])].date.nunique())))
    for shrink in (0.0,0.5,1.0):
        gs=shrink*g+(1-shrink)*gl
        vals=[]
        for lag in (357,364,371):
            k=pd.MultiIndex.from_arrays([d.station_id,d.date-pd.Timedelta(days=lag),d.hour]); vals.append(L.reindex(k).to_numpy())
        lagd=[d.date-pd.Timedelta(days=x) for x in (357,364,371)]
        Vv=np.vstack(vals)
        for i,ld in enumerate(lagd): Vv[i][ld.isin(HS).to_numpy()]=np.nan
        A=np.nanmedian(Vv,axis=0)+d.station_id.map(gs).to_numpy()
        ok=~np.isnan(A)&o&~d.date.isin(HS).to_numpy()
        b=r(P[o],y[o]); s=[f"shrink{shrink}: alone {r(A[ok],y[ok]):.4f} (v8d same rows {r(P[ok],y[ok]):.4f})"]
        for w in (0.1,0.2,0.3,0.4):
            Q=np.where(ok,(1-w)*P+w*A,P); s.append(f"w{w} {r(Q[o],y[o])-b:+.5f}")
        print(" | ".join(s))
