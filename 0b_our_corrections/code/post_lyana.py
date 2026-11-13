"""작년 같은 시기 실제(364±7일 같은 요일 3개 중앙값, log) + 역별 성장률(5~6월 2026/2025, 전체와 0.5 축소)을 w 로 혼합.
fold B: w0.1~0.2 -0.0020~-0.0025. 제외: 신규역·개통영향역, 공휴일/명절±1, 작년 공휴일 lag, 문학·부천·아시아드 주변역."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"../code"); sys.path.insert(0,"."); import sincon_v2 as V
inp,out,w=sys.argv[1],sys.argv[2],float(sys.argv[3]); HS=set(V.HSET); NEW=[3107,3108,3109]
tr=pd.read_csv("train.csv",parse_dates=["date"]); tr["lb"]=np.log1p(tr.boardings)
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id"); P=np.log1p(te.boardings.to_numpy())
geo=pd.read_csv("station_geo_graph.csv",encoding="utf-8-sig").set_index("station_id")
BIGW=set()
for a,b in [("2025-10-02","2025-10-10"),("2026-09-23","2026-09-28")]: BIGW|=set(pd.date_range(a,b))
def wd(df): return df[(df.date.dt.dayofweek<5)&~df.date.isin(HS)]
def lv(df,lo,hi): x=df[df.date.between(lo,hi)]; return x.groupby("station_id").boardings.sum()/x.date.nunique()
h=wd(tr)
g=np.log(lv(h,"2026-05-01","2026-06-30")/lv(h,"2025-05-01","2025-06-30"))
gl=np.log(h[h.date.between("2026-05-01","2026-06-30")].boardings.sum()/h[h.date.between("2026-05-01","2026-06-30")].date.nunique()/(h[h.date.between("2025-05-01","2025-06-30")].boardings.sum()/h[h.date.between("2025-05-01","2025-06-30")].date.nunique()))
gs=0.5*g+0.5*gl
MODE=sys.argv[4] if len(sys.argv)>4 else 'excl'
brk=np.log(lv(h,"2025-07-07","2025-07-25")/lv(h,"2025-06-09","2025-06-27")); brk-=brk.median()
if MODE=='brkadj': gs=gs-brk.where(brk.abs()>0.04,0)  # 개통 전후 수준 변화만큼 성장률 보정
excl=(set(brk.index[brk.abs()>0.04]) if MODE=='excl' else set())|set(NEW)|set(geo.index[(geo.dist_munhak<3000)|(geo.dist_bucheon_stadium<3000)|(geo.dist_asiad<2000)])
L=tr.set_index(["station_id","date","hour"]).lb; vals=[]
for lag in (357,364,371):
    ld=te.date-pd.Timedelta(days=lag); v=L.reindex(pd.MultiIndex.from_arrays([te.station_id,ld,te.hour])).to_numpy().copy()
    v[(ld.isin(HS)|ld.isin(BIGW)).to_numpy()]=np.nan; vals.append(v)
A=np.nanmedian(np.vstack(vals),axis=0)+te.station_id.map(gs).to_numpy()
ok=~np.isnan(A)&~te.station_id.isin(excl).to_numpy()&~te.date.isin(HS).to_numpy()&~te.date.isin(BIGW).to_numpy()
Q=np.where(ok,(1-w)*P+w*A,P)
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(np.clip(Q,0,None))}).to_csv(out,index=False)
print("mixed rows",ok.sum(),"of",len(te),"| excluded stations",len(excl),"| mean shift %+.4f rms %.4f"%((Q-P)[ok].mean(),np.sqrt(((Q-P)[ok]**2).mean())))
