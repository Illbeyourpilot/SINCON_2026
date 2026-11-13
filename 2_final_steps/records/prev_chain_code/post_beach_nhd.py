"""(1) 송도해변축제(8/8, 8/15)에 v8d 가 넣은 송도 행사 보정 제거 - 2025 축제 기간 주변역 변화 없음
   (2) 능허대 축제 2026(10/9~10, 선학) : 2025 능허대(송도) 때 행사역 +22% -> 선학 3128 에 절반(+10%)"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
os.environ["SINCON_OUT"]="../output/tmp"; sys.argv_=sys.argv[:]; sys.argv=sys.argv[:1]; sys.path.insert(0, "../code")
import final_post as F
inp,out=sys.argv_[1],sys.argv_[2]
F.EVENTS={"songdo":(["2026-08-08","2026-08-15"],0.5)}
vn=F.venue_correction(); te=F.te
s=te[["example_id"]].merge(pd.read_csv(inp),on="example_id"); p=np.log1p(s.boardings.to_numpy())
p=p-vn; print("beach rows",(vn!=0).sum(),"max %.2f"%vn.max())
m=(te.station_id.eq(3128)&te.date.isin(pd.to_datetime(["2026-10-09","2026-10-10"]))).to_numpy(); p[m]+=0.5*np.log(1.22); print("nhd rows",m.sum())
pd.DataFrame({"example_id":s.example_id,"boardings":np.expm1(np.clip(p,0,None))}).to_csv(out,index=False)
