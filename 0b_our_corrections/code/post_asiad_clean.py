"""작년 세븐틴(2025-09-13~14)·스트레이키즈(10-18~19) 공연 잔상 제거: 아시아드 2개 역 20~23시를 주변 주말 중앙값으로"""
import sys, numpy as np, pandas as pd
inp,out=sys.argv[1],sys.argv[2]
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id"); te["p"]=np.log1p(te.boardings)
bad=pd.to_datetime(["2026-09-12","2026-09-13","2026-10-17","2026-10-18"])
P=te.p.to_numpy().copy(); n=0
for d in bad:
    for st in (3209,3210):
        ref=te[(te.station_id==st)&(te.date.dt.dayofweek==d.dayofweek)&((te.date-d).abs().dt.days.between(7,35))&~te.date.isin(bad)]
        med=ref.groupby("hour").p.median()
        m=((te.date==d)&(te.station_id==st)&(te.hour>=20)).to_numpy()
        new=te.loc[m,"hour"].map(med).to_numpy(); P[m]=np.minimum(P[m],new); n+=m.sum()
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(P)}).to_csv(out,index=False)
print("rows",n,"sum %.0f -> %.0f"%(np.expm1(te.p.to_numpy())[te.date.isin(bad)&te.station_id.isin([3209,3210])&(te.hour>=20)].sum(),np.expm1(P)[(te.date.isin(bad)&te.station_id.isin([3209,3210])&(te.hour>=20)).to_numpy()].sum()))
