"""주말(비공휴일) YoY 수준 보정: adj(d)=k*log(target*작년 같은 주(±win일) 실제/올해 예측), target=2026/2025 5~6월 주말. fold B: win45 k0.5 0.17391->0.17320"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"../code"); sys.path.insert(0,"."); import sincon_v2 as V
inp,out,win,k=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4]); NEW=[3107,3108,3109]
tr=pd.read_csv("train.csv",parse_dates=["date"]); te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
def we(df): return df[~df.station_id.isin(NEW)&df.date.dt.dayofweek.isin([5,6])&~df.date.isin(V.HSET)]
act=we(tr).groupby("date").boardings.sum(); pred=we(te).groupby("date").boardings.sum()
target=act["2026-05-01":"2026-06-30"].mean()/act["2025-05-01":"2025-06-30"].mean()
adj={}
for day in pred.index:
    pw=pred[day-pd.Timedelta(days=win):day+pd.Timedelta(days=win)]
    lw=act.reindex([x-pd.Timedelta(days=364) for x in pw.index]).dropna()
    adj[day]=k*np.log(target*lw.mean()/pw[[x+pd.Timedelta(days=364) for x in lw.index]].mean())
mask=(~te.station_id.isin(NEW)&te.date.dt.dayofweek.isin([5,6])&~te.date.isin(V.HSET)).to_numpy()
p=np.log1p(te.boardings.to_numpy())+np.where(mask,te.date.map(adj).fillna(0).to_numpy(),0)
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(np.clip(p,0,None))}).to_csv(out,index=False)
s=pd.Series(adj); print("target %.4f"%target, s.groupby(s.index.month).mean().round(4).to_dict())
