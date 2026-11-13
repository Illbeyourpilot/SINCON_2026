import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
tr,_,_,_=V.load(); tr["lb"]=np.log1p(tr.boardings); HS=V.HSET
BIGW=set()  # 설/추석 연휴 블록
for a,b in [("2025-01-27","2025-01-30"),("2025-10-03","2025-10-09"),("2026-02-14","2026-02-18"),("2026-09-24","2026-09-27")]: BIGW|=set(pd.date_range(a,b))
def sunbase(df,col,day,excl):
    w=df[(df.date.dt.dayofweek==6)&((df.date-day).abs().dt.days<=21)&~df.date.isin(excl)]
    return w.groupby(["station_id","hour"])[col].median()
for f in "ABC":
    d=load_fold(f); P,_,_=v8d_blend(d); d["P"]=P; a=pd.Timestamp(V.FOLDS[f][0])
    hols=sorted(set(d.date[d.date.isin(HS)]))
    # 과거 휴일 편차 (휴일 실제 - 주변 일요일 실제), 단일휴일/명절 분리
    past=[h for h in HS if h<=a and h>=pd.Timestamp("2025-01-01")]
    def devs(days):
        out=[]
        for h in days:
            b=sunbase(tr[tr.date<=a],"lb",h,HS|BIGW); x=tr[tr.date==h].set_index(["station_id","hour"]).lb
            if len(x): out.append(x-b)
        return pd.concat(out,axis=1).mean(axis=1) if out else None
    dS=devs([h for h in past if h not in BIGW and h.dayofweek<5]); dB=devs([h for h in past if h in BIGW and h.dayofweek<5])
    rows=[]
    for h in hols:
        m=(d.date==h).to_numpy()&~d.station_id.isin([3107,3108,3109]).to_numpy()
        idx=pd.MultiIndex.from_frame(d.loc[m,["station_id","hour"]])
        S=sunbase(d,"P",h,HS|BIGW).reindex(idx).to_numpy()
        y=d.lb.to_numpy()[m]; p=P[m]
        dv=(dB if h in BIGW else dS)
        alt=S+(dv.reindex(idx).fillna(0).to_numpy() if dv is not None else 0)
        rows.append(dict(h=h.date(),big=h in BIGW,dow=h.dayofweek,n=m.sum(),P=r(p,y),SUN=r(S,y),SUNdev=r(alt,y),mix=r(0.5*p+0.5*alt,y),biasP=np.mean(p-y),biasSun=np.mean(S-y)))
    t=pd.DataFrame(rows); print("fold",f,"| past single n",sum(1 for h in past if h not in BIGW and h.dayofweek<5),"big n",sum(1 for h in past if h in BIGW and h.dayofweek<5)); print(t.round(3).to_string(index=False))
    w=t.n; print("  pooled MSE: P %.4f SUN %.4f SUNdev %.4f mix %.4f"%tuple((t[c]**2*w).sum()/w.sum() for c in ["P","SUN","SUNdev","mix"]))
