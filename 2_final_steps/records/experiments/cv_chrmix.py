import numpy as np, pandas as pd, warnings, os; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
V_=[t for t in ["chr_e3_c365","wx_chr_c365","wx_chr_cl","wx_chr_ft","wx_chr_daily","wx_chr_c182","wx_chr_la"] if all(os.path.exists(f"out/cv_{t}_{f}.csv") for f in "ABC")]
print("variants",V_)
base={};var={}
for f in "ABC":
    d=load_fold(f)
    for t in V_:
        x=pd.read_csv(f"out/cv_{t}_{f}.csv",parse_dates=["date"])[K+["pred"]].rename(columns={"pred":t}); d=d.merge(x,on=K,how="left")
    P,_,_=v8d_blend(d); y=d.lb.to_numpy(); o=~d.station_id.isin([3107,3108,3109]).to_numpy()
    base[f]=r(P[o],y[o])
    chron=0.5*d.chr_e3+0.5*d.wx_chr
    for t in V_:
        for w in (0.25,0.5):
            nc=(1-w)*chron+w*d[t]; Q=P+0.5*(nc-chron).fillna(0).to_numpy()
            var[(t,w,f)]=r(Q[o],y[o])
for t in V_:
    for w in (0.25,0.5):
        print(f"{t:14s} w{w}: "+" | ".join(f"{f} {var[(t,w,f)]-base[f]:+.5f}" for f in "ABC"), " avg %+.5f"%np.mean([var[(t,w,f)]-base[f] for f in "ABC"]))
