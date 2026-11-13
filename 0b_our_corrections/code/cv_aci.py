import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("../code/stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
tr=pd.read_csv("train.csv",parse_dates=["date"]); tr["lb"]=np.log1p(tr.boardings)
conv=set(x for a,b in [("2025-08-28","2025-08-30"),("2025-09-02","2025-09-05"),("2025-09-08","2025-09-12"),("2025-09-17","2025-09-21"),("2025-09-23","2025-09-24"),("2025-09-25","2025-09-28"),("2025-10-11","2025-10-12"),("2025-10-16","2025-10-19"),("2025-10-22","2025-10-25"),("2025-10-27","2025-10-30")] for x in pd.date_range(a,b))
EV=pd.to_datetime(["2025-01-17","2025-06-12","2025-06-26","2025-07-12","2025-09-13","2025-09-14","2025-09-20","2025-10-18","2025-10-19"])
excl=set(EV)|conv|set(pd.date_range("2025-10-17","2025-10-26"))|{pd.Timestamp("2025-09-06")}
ST=[3136,3137]; off=lambda s: s.isin(V.HSET)|(s.dt.dayofweek>=5)
def dev1(d):
    base=tr[tr.station_id.isin(ST)&((tr.date-d).abs().dt.days<=28)&(off(tr.date)==(d in V.HSET or d.dayofweek>=5))&~tr.date.isin(excl)].groupby(["station_id","hour"]).lb.median()
    return tr[(tr.date==d)&tr.station_id.isin(ST)].set_index(["station_id","hour"]).lb-base
D={d:dev1(d) for d in EV}
dA=load_fold("A"); PA,_,_=v8d_blend(dA); y=dA.lb.to_numpy()
res={k:[] for k in (0,0.25,0.5,0.75,1.0)}
for d in EV:
    if not (pd.Timestamp("2025-07-01")<=d<=pd.Timestamp("2025-10-31")): continue
    dev=pd.concat([D[x] for x in EV if x!=d],axis=1).mean(axis=1); dev[dev.index.get_level_values(1)<17]=0
    m=((dA.date==d)&dA.station_id.isin(ST)).to_numpy(); idx=pd.MultiIndex.from_frame(dA.loc[m,["station_id","hour"]])
    for k in res: res[k].append(PA[m]+k*dev.reindex(idx).fillna(0).to_numpy()-y[m])
print("fold A LOO (evening 17-23h only adj, eval all hours on event days):"," | ".join(f"k{k}: {np.sqrt((np.concatenate(v)**2).mean()):.4f}" for k,v in res.items()))
dev=pd.concat(D.values(),axis=1).mean(axis=1); print(dev.unstack(0).loc[17:23].round(2).to_string())
