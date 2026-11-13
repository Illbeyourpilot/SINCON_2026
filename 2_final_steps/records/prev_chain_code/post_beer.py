"""송도맥주축제 2026(8/22~30, 유료화로 관람객 약 0.45배) 20~23시 재보정.
   추가 승차(인원) = 0.45 x 2025 같은 유형 날의 (실제 - 기준선). 유형: 토, 일, 평일, 마지막날(2025-08-30).
   새 값 = 2026 기준선(축제 없는 주변 같은 요일 예측 중앙값) + 추가 승차. 기존 축제 효과를 대체하므로 중복 없음."""
import sys, numpy as np, pandas as pd
inp,out,ratio=sys.argv[1],sys.argv[2],float(sys.argv[3])
S=[3138,3139]; H=[20,21,22,23]
tr=pd.read_csv("train.csv",parse_dates=["date"]); te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
FEST25=pd.date_range("2025-08-22","2025-08-30"); FEST26=pd.date_range("2026-08-22","2026-08-30")
EXCL=set(FEST25)|set(FEST26)|set(pd.date_range("2025-07-31","2025-08-03"))|set(pd.date_range("2026-07-31","2026-08-02"))|set(pd.to_datetime(["2025-08-15","2026-08-15","2026-08-17","2026-09-12"]))
def base(df,col,d):
    same=(df.date.dt.dayofweek>=5)==(d.dayofweek>=5) if d.dayofweek<5 else (df.date.dt.dayofweek==d.dayofweek)
    m=df.station_id.isin(S)&df.hour.isin(H)&((df.date-d).abs().dt.days<=28)&same&~df.date.isin(EXCL)
    return df[m].groupby(["station_id","hour"])[col].median()
def kind(d,last): return "last" if d==last else ("sat" if d.dayofweek==5 else "sun" if d.dayofweek==6 else "wd")
ex={}
for d in FEST25:
    k=kind(d,FEST25[-1]); x=tr[(tr.date==d)&tr.station_id.isin(S)&tr.hour.isin(H)].set_index(["station_id","hour"]).boardings
    ex.setdefault(k,[]).append(x-base(tr,"boardings",d))
EX={k:pd.concat(v,axis=1).mean(axis=1) for k,v in ex.items()}
b=te.boardings.to_numpy().copy()
for d in FEST26:
    k=kind(d,FEST26[-1]); bl=base(te,"boardings",d)
    m=((te.date==d)&te.station_id.isin(S)&te.hour.isin(H)).to_numpy(); ix=pd.MultiIndex.from_frame(te.loc[m,["station_id","hour"]])
    new=bl.reindex(ix).to_numpy()+ratio*EX[k].reindex(ix).fillna(0).to_numpy()
    old=b[m].copy(); b[m]=np.where(np.isnan(new),b[m],np.clip(new,1,None))
    h22=(te.loc[m,"hour"].to_numpy()==22)&(te.loc[m,"station_id"].to_numpy()==3139)
    print(d.strftime("%m-%d %a"),k,"3139 22h %.0f -> %.0f"%(old[h22][0],b[m][h22][0]))
pd.DataFrame({"example_id":te.example_id,"boardings":b}).to_csv(out,index=False)
