import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("../code/stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
tr=pd.read_csv("train.csv",parse_dates=["date"]); tr["lb"]=np.log1p(tr.boardings)
geo=pd.read_csv("station_geo_graph.csv").set_index("station_id"); near=geo.index[geo.dist_bucheon_stadium<3000].tolist()
g=pd.read_csv("bucheon_home_actual.csv",parse_dates=["date"]).set_index("date").start_hour
G26=[d for d in g.index if pd.Timestamp("2026-03-01")<=d<=pd.Timestamp("2026-06-30")]
off=lambda s: s.isin(V.HSET)|(s.dt.dayofweek>=5)
def base(df,col,d):
    m=df.station_id.isin(near)&((df.date-d).abs().dt.days<=21)&(off(df.date)==(d in V.HSET or d.dayofweek>=5))&~df.date.isin(g.index)
    return df[m].groupby(["station_id","hour"])[col].median()
def devrel(d):  # 편차를 경기 시작 기준 상대시간으로
    x=tr[(tr.date==d)&tr.station_id.isin(near)].set_index(["station_id","hour"]).lb-base(tr,"lb",d)
    x=x.rename("dv").reset_index(); x["rel"]=x.hour-g[d]; return x.set_index(["station_id","rel"]).dv
D={d:devrel(d) for d in G26}
dB=load_fold("B"); P,_,_=v8d_blend(dB); y=dB.lb.to_numpy()
res={k:[] for k in ("v8d",0.5,0.75,1.0)}
for d in G26:
    dev=pd.concat([D[x] for x in G26 if x!=d],axis=1).mean(axis=1)
    m=((dB.date==d)&dB.station_id.isin(near)&(dB.hour-g[d]).between(-3,6)).to_numpy()
    sub=dB.loc[m,["station_id","hour"]].copy(); sub["rel"]=sub.hour-g[d]
    b=base(dB.assign(p=P),"p",d).reindex(pd.MultiIndex.from_frame(sub[["station_id","hour"]])).to_numpy()
    dv=dev.reindex(pd.MultiIndex.from_frame(sub[["station_id","rel"]])).fillna(0).to_numpy()
    res["v8d"].append(P[m]-y[m])
    for k in (0.5,0.75,1.0): res[k].append(np.where(np.isnan(b),P[m],b+k*dv)-y[m])
print("fold B LOO on 2026 Bucheon games (near stations, rel -3..+6):"," | ".join(f"{k}: {np.sqrt((np.concatenate(v)**2).mean()):.4f}" for k,v in res.items()))
print("stations",near)
# --- 관중 수 활용: 선형(인원) 공간에서 추가 승차 = 관중 x 비율 ---
att=pd.read_csv("bucheon_att.csv",parse_dates=["date"]).set_index("date").att
def extra(d):  # 실제 - 기준선 (인원), 상대시간
    b=np.expm1(base(tr,"lb",d)); x=tr[(tr.date==d)&tr.station_id.isin(near)].set_index(["station_id","hour"]).boardings
    e=(x-b).rename("e").reset_index(); e["rel"]=e.hour-g[d]; return e.set_index(["station_id","rel"]).e
E={d:extra(d) for d in G26}
for mode in ("rate_att","rate_flat","logdev_att"):
    errs=[]
    for d in G26:
        others=[x for x in G26 if x!=d]
        m=((dB.date==d)&dB.station_id.isin(near)&(dB.hour-g[d]).between(-3,6)).to_numpy()
        sub=dB.loc[m,["station_id","hour"]].copy(); sub["rel"]=sub.hour-g[d]
        bl=base(dB.assign(p=P),"p",d).reindex(pd.MultiIndex.from_frame(sub[["station_id","hour"]])).to_numpy()
        ix=pd.MultiIndex.from_frame(sub[["station_id","rel"]])
        if mode=="rate_att":
            rate=pd.concat([E[x]/att[x] for x in others],axis=1).mean(axis=1); add=att[d]*rate.reindex(ix).fillna(0).to_numpy()
            new=np.log1p(np.clip(np.expm1(bl)+add,0,None))
        elif mode=="rate_flat":
            ex=pd.concat([E[x] for x in others],axis=1).mean(axis=1); new=np.log1p(np.clip(np.expm1(bl)+ex.reindex(ix).fillna(0).to_numpy(),0,None))
        else:
            dv=pd.concat([D[x] for x in others],axis=1).mean(axis=1); sc=np.log(att[d])/np.mean([np.log(att[x]) for x in others])
            new=bl+dv.reindex(ix).fillna(0).to_numpy()*sc
        errs.append(np.where(np.isnan(bl),P[m],new)-y[m])
    print(mode,"%.4f"%np.sqrt((np.concatenate(errs)**2).mean()))
