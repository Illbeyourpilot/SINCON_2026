import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
tr,_,_,geo=V.load(); tr["lb"]=np.log1p(tr.boardings); HS=set(V.HSET); geo=geo.set_index("station_id")
f="B"; d=load_fold(f); P,_,_=v8d_blend(d); y=d.lb.to_numpy(); a=pd.Timestamp(V.FOLDS[f][0])
o=~d.station_id.isin([3107,3108,3109]).to_numpy(); hol=d.date.isin(HS).to_numpy()
L=tr.set_index(["station_id","date","hour"]).lb
def wd(df): return df[(df.date.dt.dayofweek<5)&~df.date.isin(HS)]
def lv(df,lo,hi): x=df[df.date.between(lo,hi)]; return x.groupby("station_id").boardings.sum()/x.date.nunique()
h=wd(tr); g=np.log(lv(h,"2026-01-01","2026-02-28")/lv(h,"2025-01-01","2025-02-28"))
gl=np.log(h[h.date.between("2026-01-01","2026-02-28")].boardings.mean()/h[h.date.between("2025-01-01","2025-02-28")].boardings.mean()); gs=0.5*g+0.5*gl
def lagmed(lags):
    vals=[]
    for lag in lags:
        ld=d.date-pd.Timedelta(days=lag); v=L.reindex(pd.MultiIndex.from_arrays([d.station_id,ld,d.hour])).to_numpy().copy()
        v[ld.isin(HS).to_numpy()]=np.nan; vals.append(v)
    return np.nanmedian(np.vstack(vals),axis=0)
A3=lagmed((357,364,371))+d.station_id.map(gs).to_numpy()
A5=lagmed((350,357,364,371,378))+d.station_id.map(gs).to_numpy()
# delta 방식: 최근 28일 수준(역x요일유형x시간 중앙값) + (작년 t 값 - 작년 기준기간 수준)
def dt(s): return np.where(s.dt.dayofweek==5,1,np.where(s.dt.dayofweek==6,2,0))
def lvlh(lo,hi):
    x=tr[tr.date.between(lo,hi)&~tr.date.isin(HS)].copy(); x["t"]=dt(x.date); return x.groupby(["station_id","t","hour"]).lb.median()
k=pd.MultiIndex.from_arrays([d.station_id,dt(d.date),d.hour])
cur=lvlh(a-pd.Timedelta(days=27),a).reindex(k).to_numpy(); ly=lvlh(a-pd.Timedelta(days=27+364),a-pd.Timedelta(days=364)).reindex(k).to_numpy()
D3=cur+lagmed((357,364,371))-ly
ssg=set(pd.read_csv("ssg_home_actual.csv",parse_dates=["date"]).date); bfc=set(pd.read_csv("bucheon_home_actual.csv",parse_dates=["date"]).date)
ev=set(geo.index[(geo.dist_munhak<3000)|(geo.dist_bucheon_stadium<3000)|(geo.dist_asiad<2000)])
evday=(d.date.isin(ssg)|d.date.isin(bfc)|(d.date-pd.Timedelta(days=364)).isin(ssg)|(d.date-pd.Timedelta(days=364)).isin(bfc)).to_numpy()
evst=d.station_id.isin(ev).to_numpy()
b=r(P[o],y[o])
for nm,A in [("A3",A3),("A5",A5),("D3",D3),("avg A3+D3",(A3+D3)/2)]:
    for ex,msk in [("excl ev stations",~evst),("excl ev days only",~(evst&evday)),("no excl",np.ones(len(d),bool))]:
        ok=~np.isnan(A)&o&~hol&msk; s=[f"{nm:10s} {ex:18s} alone {r(A[ok],y[ok]):.4f}"]
        for w in (0.1,0.15,0.2,0.25): Q=np.where(ok,(1-w)*P+w*A,P); s.append(f"w{w} {r(Q[o],y[o])-b:+.5f}")
        print(" | ".join(s))
