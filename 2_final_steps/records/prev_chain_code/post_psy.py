"""펜타포트 보정: 2025(8/1~3 금토일) 시간대별 편차를 2026(7/31~8/2 금토일) 주변역에 적용. new = 주변일 기준선 + w*편차"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "../code"); import sincon_v2 as V
inp, out, w = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv)>3 else 1.0
tr=pd.read_csv("train.csv",parse_dates=["date"]); tr["lb"]=np.log1p(tr.boardings)
te=pd.read_csv("test.csv",parse_dates=["date"]); s=pd.read_csv(inp)
te=te.merge(s,on="example_id"); te["p"]=np.log1p(te.boardings)
geo=pd.read_csv("station_geo_graph.csv").set_index("station_id")
la,lo=geo.loc[3209,"lat"],geo.loc[3209,"lon"]
near=[x for x in geo.index if V.hav(geo.loc[x,"lat"],geo.loc[x,"lon"],la,lo)<2000]
EV=pd.to_datetime(["2025-06-28","2026-07-11","2025-09-13","2025-09-14","2025-10-18","2025-10-19","2026-04-04","2026-04-05","2026-06-20","2026-06-21","2025-08-01","2025-08-02","2025-08-03","2025-08-22","2025-08-23","2025-08-24","2025-08-25","2025-08-26",
  "2025-08-27","2025-08-28","2025-08-29","2025-08-30","2026-07-31","2026-08-01","2026-08-02","2026-08-08","2026-08-15",
  "2026-08-22","2026-08-23","2026-08-24","2026-08-25","2026-08-26","2026-08-27","2026-08-28","2026-08-29","2026-08-30","2026-09-12"])
def base(df,col,day):
    wd=day.dayofweek>=5
    m=df.station_id.isin(near)&((df.date-day).abs().dt.days<=21)&((df.date.dt.dayofweek>=5)==wd)&~df.date.isin(EV)&~df.date.isin(V.HSET)
    return df[m].groupby(["station_id","hour"])[col].median()
P=te.p.to_numpy().copy(); n=0
for d25,d26 in zip(pd.to_datetime(["2025-06-28"]),pd.to_datetime(["2026-07-11"])):
    x=tr[(tr.date==d25)&tr.station_id.isin(near)].set_index(["station_id","hour"]).lb
    dev=(x-base(tr,"lb",d25)).dropna()
    keep=dev.groupby(level=0).max(); keep=keep[keep>0.3].index
    b26=base(te,"p",d26)
    m=((te.date==d26)&te.station_id.isin(keep)).to_numpy()
    idx=pd.MultiIndex.from_frame(te.loc[m,["station_id","hour"]])
    nv=b26.reindex(idx).to_numpy()+w*dev.reindex(idx).fillna(0).to_numpy()
    P[m]=np.where(np.isnan(nv),P[m],nv); n+=m.sum()
    print(d26.date(),"stations",list(keep),"pred sum %.0f -> %.0f"%(np.expm1(te.p.to_numpy()[m]).sum(),np.expm1(P[m]).sum()),
          "| 2025 actual %.0f"%np.expm1(x.reindex(idx)).sum())
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(np.clip(P,0,None))}).to_csv(out,index=False)
print("rows",n)
