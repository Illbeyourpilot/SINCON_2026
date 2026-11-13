import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
exec(open("cv_holsun.py",encoding="utf-8").read().split("for f in \"ABC\":")[0])
for f in "ABC":
    d=load_fold(f); P,_,_=v8d_blend(d); a=pd.Timestamp(V.FOLDS[f][0]); y=d.lb.to_numpy(); o=~d.station_id.isin([3107,3108,3109]).to_numpy()
    past=[h for h in HS if pd.Timestamp("2025-01-01")<=h<=a]
    def devs(days):
        out=[]
        for h in days:
            b=sunbase(tr[tr.date<=a],"lb",h,HS|BIGW); x=tr[tr.date==h].set_index(["station_id","hour"]).lb
            if len(x): out.append(x-b)
        return pd.concat(out,axis=1).mean(axis=1) if out else None
    dS=devs([h for h in past if h not in BIGW and h.dayofweek<5]); dB=devs([h for h in past if h in BIGW and h.dayofweek<5])
    ALT=np.full(len(d),np.nan)
    for h in sorted(set(d.date[d.date.isin(HS)])):
        m=(d.date==h).to_numpy(); idx=pd.MultiIndex.from_frame(d.loc[m,["station_id","hour"]])
        dv=dB if h in BIGW else dS
        ALT[m]=sunbase(d.assign(P=P),"P",h,HS|BIGW).reindex(idx).to_numpy()+(dv.reindex(idx).fillna(0).to_numpy() if dv is not None else 0)
    sat=(d.date.dt.dayofweek==5).to_numpy(); ok=~np.isnan(ALT)&o
    base=r(P[o],y[o]); out=[]
    for w in (0.3,0.4,0.5,0.6):
        for nosat in (False,True):
            m=ok&(~sat if nosat else True); Q=P.copy(); Q[m]=(1-w)*P[m]+w*ALT[m]
            out.append(f"w{w}{'_nosat' if nosat else ''} {r(Q[o],y[o])-base:+.5f}")
    print(f, "base %.5f |"%base, " ".join(out))
