"""부천FC(K리그1 승격) 홈경기 보정: 2026 상반기 8경기의 경기시작 기준 시간대별 편차로 주변 4역(3753~3756) 대체. fold B LOO 0.352->0.284"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "../code"); import sincon_v2 as V
inp,out,k=sys.argv[1],sys.argv[2],float(sys.argv[3])
src=open("../code/cv_bucheon.py",encoding="utf-8").read().split("dB=load_fold")[0]
src=src.replace('exec(open("../code/stack_cv.py", encoding="utf-8").read().split("if __name__")[0])',"")
exec(src)
dev=pd.concat(D.values(),axis=1).mean(axis=1)
att=pd.read_csv("bucheon_att.csv",parse_dates=["date"]).set_index("date").att
ml=np.mean([np.log(att[x]) for x in G26])
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id"); te["p"]=np.log1p(te.boardings)
P=te.p.to_numpy().copy()
for d in [x for x in g.index if x>pd.Timestamp("2026-06-30")]:
    m=((te.date==d)&te.station_id.isin(near)&(te.hour-g[d]).between(-3,6)).to_numpy()
    sub=te.loc[m,["station_id","hour"]].copy(); sub["rel"]=sub.hour-g[d]
    b=base(te,"p",d).reindex(pd.MultiIndex.from_frame(sub[["station_id","hour"]])).to_numpy()
    dv=dev.reindex(pd.MultiIndex.from_frame(sub[["station_id","rel"]])).fillna(0).to_numpy()
    sc=np.log(att[d])/ml; P[m]=np.where(np.isnan(b),P[m],b+k*sc*dv)
    pk=(te.station_id==3754).to_numpy()&m&(te.hour==g[d]+2).to_numpy()
    print(d.date(),"att",att[d],"start",g[d],"3754 end-hour %.0f -> %.0f"%(te.boardings.to_numpy()[pk][0],np.expm1(P[pk])[0]))
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(np.clip(P,0,None))}).to_csv(out,index=False)
