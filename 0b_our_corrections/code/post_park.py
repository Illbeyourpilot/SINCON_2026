"""인천대공원 권역 가을 공휴일 나들이 보정: 3226 x1.44, 3227 x1.12 (CV 공휴일 8회 LOO 검증) 의 k 제곱. 대상 10/3, 10/5, 10/9 (추석 핵심일·여름 공휴일 제외)"""
import sys, numpy as np, pandas as pd
inp,out,k=sys.argv[1],sys.argv[2],float(sys.argv[3])
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
p=np.log1p(te.boardings.to_numpy()); days=pd.to_datetime(["2026-10-03","2026-10-05","2026-10-09"])
for st,f in ((3226,1.44),(3227,1.12)):
    m=(te.station_id.eq(st)&te.date.isin(days)).to_numpy(); p[m]+=k*np.log(f)
    print(st,"rows",m.sum(),"x%.2f"%(f**k))
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(p)}).to_csv(out,index=False)
