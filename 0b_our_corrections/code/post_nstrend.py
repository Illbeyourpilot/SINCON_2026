"""신규역 감쇠 추세 보정: 최근 3개월 기존역 대비 수준의 월 성장률 g 를 phi=0.8 로 감쇠 연장, 월별 예측 수준을 목표로 맞춤"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../code"); import sincon_v2 as V
inp,out,phi=sys.argv[1],sys.argv[2],float(sys.argv[3]); NEW=[3107,3108,3109]
tr=pd.read_csv("train.csv",parse_dates=["date"]); te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
def ratio(df):
    x=df[(df.date.dt.dayofweek<5)&~df.date.isin(V.HSET)].copy(); x["m"]=x.date.dt.to_period("M")
    o=x[~x.station_id.isin(NEW)].groupby("m").boardings.sum()/x[~x.station_id.isin(NEW)].groupby("m").date.nunique()
    return pd.DataFrame({s:(x[x.station_id==s].groupby("m").boardings.sum()/x[x.station_id==s].groupby("m").date.nunique())/o for s in NEW})
hist=ratio(tr[tr.date>="2026-04-01"]); g=np.log(hist.iloc[-1]/hist.iloc[0])/2; pr=ratio(te)
p=np.log1p(te.boardings.to_numpy())
for s in NEW:
    for i,mth in enumerate(pr.index,1):
        tgt=np.log(hist[s].iloc[-1])+g[s]*sum(phi**j for j in range(1,i+1)); adj=tgt-np.log(pr.loc[mth,s])
        m=((te.station_id==s)&(te.date.dt.to_period("M")==mth)).to_numpy(); p[m]+=adj
        print(s,mth,"g %.3f"%g[s],"adj %+.3f"%adj)
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(np.clip(p,0,None))}).to_csv(out,index=False)
