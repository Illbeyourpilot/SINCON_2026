import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
T="C:/sincon/teamcv/"; CA=pd.read_pickle(T+"cv_all.pkl")
def rd(p,col,name): x=pd.read_csv(p,parse_dates=["date"]); return x[K+[col]].rename(columns={col:name})
DATA={}
for f in "AB":
    d=pd.read_parquet(f"out/post_{f}.parquet")[K+["lb","wx","game","htg"]]
    c=CA[f].copy(); c["date"]=pd.to_datetime(c.date); d=d.merge(c.drop(columns=["lb"]),on=K,how="left")
    d=d.merge(rd(T+f"results_v10/results_v10/cv_lgbL_{f}.csv","pred","lgbL"),on=K,how="left")
    d=d.merge(rd(T+f"results_v10/results_v10/cv_mlp_{f}.csv","pred","mlp"),on=K,how="left")
    d=d.merge(rd(T+f"cv_folds_v8dh13_v8dh17/cv_v8dh13_{f}.csv","pred_log1p","h13"),on=K,how="left")
    d=d.merge(rd(T+f"cv_folds_v8dh13_v8dh17/cv_v8dh17_{f}.csv","pred_log1p","h17"),on=K,how="left")
    def post(core):
        P=core+d.wx+0.4*d.game; h=d.htg.to_numpy()
        return np.clip(np.where(np.isnan(h),P,0.5*P+0.5*np.nan_to_num(h)),0,None)
    d["v8d"]=post(0.5*(0.75*(d.lt1+d.lt2)/2+0.25*(d.v3_x10+d.v3_x8)/2)+0.5*(d.chr_e3+d.wx_chr)/2)
    d["v10"]=post(0.8*(0.5*(d.lt1+d.lt2)/2+0.5*(d.chr_e3+d.wx_chr_st)/2)+0.2*d.lgbL)
    d["tfm_p"]=post(d.tfm); d["mlp_p"]=post(d.mlp)
    DATA[f]=d
    print(f,"missing:",{c:int(d[c].isna().sum()) for c in ["lt1","lgbL","h13","h17","mlp","tfm"]})
import pickle; pickle.dump(DATA,open("team_cv.pkl","wb"))
for f,d in DATA.items():
    y=d.lb.to_numpy()
    print(f," ".join(f"{c} {r(d[c].to_numpy(),y):.4f}" for c in ["v8d","v10","h13","h17","tfm_p","mlp_p"]))
