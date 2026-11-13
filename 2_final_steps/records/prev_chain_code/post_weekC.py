"""주간 YoY 평일 수준 보정 (월 단위 C 를 ±win일 창으로 세분): adj(d)=k*log(target*작년같은주실제/올해예측)"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../code"); import sincon_v2 as V
inp,out,win,k=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4])
NEW=[3107,3108,3109]
tr=pd.read_csv("train.csv",parse_dates=["date"]); te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
def wd(df): return df[~df.station_id.isin(NEW)&(df.date.dt.dayofweek<5)&~df.date.isin(V.HSET)]
act=wd(tr).groupby("date").boardings.sum()
target=act["2026-05-01":"2026-06-30"].mean()/act["2025-05-01":"2025-06-30"].mean()
pred=wd(te).groupby("date").boardings.sum()
adj={}
for day in pred.index:
    pw=pred[day-pd.Timedelta(days=win):day+pd.Timedelta(days=win)]
    ly=[x-pd.Timedelta(days=364) for x in pw.index]; lw=act.reindex(ly)
    ok=lw.notna().to_numpy()
    adj[day]=k*np.log(target*lw[ok].mean()/pw[ok].mean())
mask=(te.date<=pd.Timestamp("2026-08-31")).to_numpy()&(~te.station_id.isin(NEW)&(te.date.dt.dayofweek<5)&~te.date.isin(V.HSET)).to_numpy()
a=te.date.map(adj).fillna(0).to_numpy()
p=np.log1p(te.boardings.to_numpy())+np.where(mask,a,0)
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(np.clip(p,0,None))}).to_csv(out,index=False)
s=pd.Series(adj); print("target %.4f"%target); print(s.groupby(s.index.to_period("W")).mean().round(3).to_string())
