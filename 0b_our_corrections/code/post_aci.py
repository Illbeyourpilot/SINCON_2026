"""아트센터인천·트라이보울·센트럴파크 저녁 공연 보정 (3136,3137 17~23시): 2025 공연 9회 평균 편차 x k (LOO k0.5~0.75 최적)"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("../code/cv_aci.py",encoding="utf-8").read().split("dA=load_fold")[0].split("exec(open")[0])
sys.path.insert(0, "../code"); import sincon_v2 as V
inp,out,k=sys.argv[1],sys.argv[2],float(sys.argv[3])
src=open("../code/cv_aci.py",encoding="utf-8").read()
ns={}; exec(src.split("dA=load_fold")[0].replace('exec(open("../code/stack_cv.py", encoding="utf-8").read().split("if __name__")[0])',"import sincon_v2 as V"),ns)
dev=pd.concat(ns["D"].values(),axis=1).mean(axis=1); dev[dev.index.get_level_values(1)<17]=0; dev=dev.clip(lower=0)
days=pd.to_datetime(["2026-09-19","2026-09-20","2026-09-29","2026-10-23","2026-10-24","2026-10-25"])
te=pd.read_csv("test.csv",parse_dates=["date"]).merge(pd.read_csv(inp),on="example_id")
m=(te.date.isin(days)&te.station_id.isin([3136,3137])).to_numpy()
idx=pd.MultiIndex.from_frame(te.loc[m,["station_id","hour"]])
p=np.log1p(te.boardings.to_numpy()); a=k*dev.reindex(idx).fillna(0).to_numpy(); p[m]+=a
pd.DataFrame({"example_id":te.example_id,"boardings":np.expm1(p)}).to_csv(out,index=False)
print("rows",(a>0).sum(),"mean adj on evening rows %.3f"%a[a>0].mean())
