"""post_lyana 확장: 평일/주말 비율, lag 집합, 성장률 축소, 경기장역 처리 선택. 
사용: python post_lyana2.py in out w_wd w_we lags(357,364,371) shrink(0.5) evmode(st|day)"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"../code"); import sincon_v2 as V
inp,out=sys.argv[1],sys.argv[2]; wwd,wwe=float(sys.argv[3]),float(sys.argv[4]); LAGS=[int(x) for x in sys.argv[5].split(",")]; sh=float(sys.argv[6]); evmode=sys.argv[7]
HS=set(V.HSET); NEW=[3107,3108,3109]
tr=pd.read_csv("train.csv",parse_dates=["date"]); tr["lb"]=np.log1p(tr.boardings)
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id"); P=np.log1p(te.boardings.to_numpy())
geo=pd.read_csv("station_geo_graph.csv",encoding="utf-8-sig").set_index("station_id")
BIGW=set()
for a,b in [("2025-10-02","2025-10-10"),("2026-09-23","2026-09-28")]: BIGW|=set(pd.date_range(a,b))
def wd(df): return df[(df.date.dt.dayofweek<5)&~df.date.isin(HS)]
def lv(df,lo,hi): x=df[df.date.between(lo,hi)]; return x.groupby("station_id").boardings.sum()/x.date.nunique()
h=wd(tr); g=np.log(lv(h,"2026-05-01","2026-06-30")/lv(h,"2025-05-01","2025-06-30"))
gl=np.log(h[h.date.between("2026-05-01","2026-06-30")].boardings.sum()/h[h.date.between("2026-05-01","2026-06-30")].date.nunique()/(h[h.date.between("2025-05-01","2025-06-30")].boardings.sum()/h[h.date.between("2025-05-01","2025-06-30")].date.nunique()))
brk=np.log(lv(h,"2025-07-07","2025-07-25")/lv(h,"2025-06-09","2025-06-27")); brk-=brk.median()
gs=sh*g+(1-sh)*gl-brk.where(brk.abs()>0.04,0)
ev=set(geo.index[(geo.dist_munhak<3000)|(geo.dist_bucheon_stadium<3000)|(geo.dist_asiad<2000)])
L=tr.set_index(["station_id","date","hour"]).lb; vals=[]
for lag in LAGS:
    ld=te.date-pd.Timedelta(days=lag); v=L.reindex(pd.MultiIndex.from_arrays([te.station_id,ld,te.hour])).to_numpy().copy()
    v[(ld.isin(HS)|ld.isin(BIGW)).to_numpy()]=np.nan; vals.append(v)
A=np.nanmedian(np.vstack(vals),axis=0)+te.station_id.map(gs).to_numpy()
ok=~np.isnan(A)&~te.station_id.isin(NEW).to_numpy()&~te.date.isin(HS|BIGW).to_numpy()
evst=te.station_id.isin(ev).to_numpy()
if evmode=="st": ok&=~evst
else:
    ssg=set(pd.read_csv("ssg_home_actual.csv",parse_dates=["date"]).date); bfc=set(pd.read_csv("bucheon_home_actual.csv",parse_dates=["date"]).date)
    gd=te.date.isin(ssg|bfc).to_numpy()|np.any([(te.date-pd.Timedelta(days=l)).isin(ssg|bfc).to_numpy() for l in LAGS],axis=0)
    EM=pd.read_csv("evmask.csv").set_index("example_id").ev.reindex(te.example_id).fillna(0).to_numpy().astype(bool)
    ok&=~(evst&gd)&~EM
w=np.where(te.date.dt.dayofweek.to_numpy()>=5,wwe,wwd)
Q=np.where(ok,(1-w)*P+w*A,P)
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(np.clip(Q,0,None))}).to_csv(out,index=False)
