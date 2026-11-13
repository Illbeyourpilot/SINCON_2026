"""부평삼거리(3122, 인천가족공원 성묘) 추석 당일 9/25 : 2025 추석 당일·2026 설 당일 시간대별 log 평균과 현재 예측의 중간 (k=0.5)"""
import sys, numpy as np, pandas as pd
inp,out,k=sys.argv[1],sys.argv[2],float(sys.argv[3])
tr=pd.read_csv("train.csv",parse_dates=["date"]); tr["lb"]=np.log1p(tr.boardings)
an=tr[(tr.station_id==3122)&tr.date.isin(pd.to_datetime(["2025-10-06","2026-02-17"]))].groupby("hour").lb.mean()
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
p=np.log1p(te.boardings.to_numpy()); m=((te.station_id==3122)&(te.date=="2026-09-25")).to_numpy()
before=te.boardings[m].sum(); p[m]=(1-k)*p[m]+k*te.loc[m,"hour"].map(an).to_numpy()
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(p)}).to_csv(out,index=False)
print("rows",m.sum(),"day sum %.0f -> %.0f (analog days 9274, 11075)"%(before,np.expm1(p[m]).sum()))
