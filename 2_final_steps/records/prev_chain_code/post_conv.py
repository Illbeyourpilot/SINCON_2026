"""송도컨벤시아(인천대입구 3136) 전시·박람회일 보정: 2025 비공연 행사 6건 시간대별 편차 x k (LOO 검증 k=0.5 최적). 학술대회·컨퍼런스 제외"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../code"); import sincon_v2 as V
inp,out,k=sys.argv[1],sys.argv[2],float(sys.argv[3])
tr=pd.read_csv("train.csv",parse_dates=["date"]); tr["lb"]=np.log1p(tr.boardings)
EVS=[("2025-08-28","2025-08-30"),("2025-09-02","2025-09-05"),("2025-09-08","2025-09-12"),("2025-09-17","2025-09-21"),("2025-09-23","2025-09-24"),("2025-09-25","2025-09-28"),("2025-10-16","2025-10-17"),("2025-10-22","2025-10-25"),("2025-10-27","2025-10-30")]
ALLH=set(x for a,b in EVS+[("2025-10-11","2025-10-12"),("2025-10-17","2025-10-19"),("2025-04-18","2025-04-20"),("2025-06-14","2025-06-15"),("2025-02-14","2025-02-16"),("2025-11-04","2025-11-07"),("2025-12-04","2025-12-07"),("2026-03-25","2026-03-29"),("2026-05-29","2026-05-31")] for x in pd.date_range(a,b))
off=lambda s: s.isin(V.HSET)|(s.dt.dayofweek>=5)
devs=[]
for a,b in EVS:
    for d in pd.date_range(a,b):
        base=tr[(tr.station_id==3136)&((tr.date-d).abs().dt.days<=28)&(off(tr.date)==(d in V.HSET or d.dayofweek>=5))&~tr.date.isin(ALLH)].groupby("hour").lb.median()
        devs.append(tr[(tr.date==d)&(tr.station_id==3136)].set_index("hour").lb-base)
dev=pd.concat(devs,axis=1).mean(axis=1)
ev=pd.read_csv("conv26.csv"); ev=ev[~ev.event_type.isin(["국제 학술대회","국제 컨퍼런스","산업 컨퍼런스"])]
extra=[("2026-07-01","2026-07-04"),("2026-08-06","2026-08-08"),("2026-10-30","2026-10-31")]
days=sorted(set(x for _,r in ev.iterrows() for x in pd.date_range(r.start_date,r.end_date) if x<=pd.Timestamp("2026-10-31"))|set(x for a,b in extra for x in pd.date_range(a,b)))
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
m=(te.station_id.eq(3136)&te.date.isin(days)).to_numpy()
p=np.log1p(te.boardings.to_numpy()); p[m]+=k*te.loc[m,"hour"].map(dev).fillna(0).to_numpy()
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(p)}).to_csv(out,index=False)
print("event days",len(days),"rows",m.sum(),"mean adj %.3f"%(k*te.loc[m,"hour"].map(dev).mean()))
