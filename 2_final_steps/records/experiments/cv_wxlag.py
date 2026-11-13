import numpy as np, pandas as pd, warnings, pickle; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
w=pd.read_csv("weather_incheon_hourly.csv",parse_dates=["date"]); day=w[w.hour.between(6,23)].groupby("date").precipitation.sum()
day=day.reindex(pd.date_range(day.index.min(),day.index.max())).fillna(0)
F=pd.DataFrame({"r0":day,"r1":day.shift(1),"r3":day.shift(1)+day.shift(2)+day.shift(3)})
wet=(day>=5).astype(int); run=wet.groupby((wet!=wet.shift()).cumsum()).cumsum()*wet
F["run_prev"]=run.shift(1); F["after_rain"]=((F.r1>=10)&(F.r0<1)).astype(int)
def bins(d):
    x=d.join(F,on="date")
    x["b_r1"]=pd.cut(x.r1,[-1,1,10,30,1000],labels=["<1","1-10","10-30","30+"])
    x["b_r3"]=pd.cut(x.r3,[-1,1,20,60,1000],labels=["<1","1-20","20-60","60+"])
    return x
HS=V.HSET; out={}
DATA=pickle.load(open("team_cv.pkl","rb"))
for f in "ABC":
    d=load_fold(f); P,_,_=v8d_blend(d); d["e"]=d.lb-P; d=d[~d.station_id.isin([3107,3108,3109])]
    d["t"]=np.where(d.date.isin(HS)|(d.date.dt.dayofweek>=5),"off","wd")
    srcs=[("v8d",d)]
    if f in DATA:
        h=DATA[f][K+["lb","h17"]].dropna(); h["e"]=h.lb-h.h17; h["t"]=np.where(h.date.isin(HS)|(h.date.dt.dayofweek>=5),"off","wd"); srcs.append(("h17",h))
    for nm,x in srcs:
        x=bins(x); x=x[x.r0<1]  # 당일 비 없는 날만: 순수 지연효과
        for col in ["b_r1","b_r3"]:
            g=x.groupby(["t",col]).e.agg(["mean","count"]); g["mean"]=g["mean"]-x.groupby("t").e.mean().reindex(g.index.get_level_values(0)).to_numpy()
            out[(f,nm,col)]=g
        a=x[x.after_rain==1]; print(f,nm,"after_rain(어제>=10mm, 오늘<1mm) days",a.date.nunique(),"rel resid wd %+.4f off %+.4f"%(
            a[a.t=="wd"].e.mean()-x[x.t=="wd"].e.mean() if (a.t=="wd").any() else np.nan, a[a.t=="off"].e.mean()-x[x.t=="off"].e.mean() if (a.t=="off").any() else np.nan))
for col in ["b_r1","b_r3"]:
    print("\n== 오늘 비 없는 날, 잔차(실제-예측) 상대값 by",col)
    t=pd.concat({f"{f}_{nm}":out[(f,nm,c)]["mean"] for (f,nm,c) in out if c==col},axis=1)
    print(t.round(4).to_string())
