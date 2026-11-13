import pandas as pd, numpy as np, sys
D=pd.read_pickle('cv_all.pkl'); K=['date','station_id','hour']
folds=sys.argv[1] if len(sys.argv)>1 else 'AB'
r=lambda d,p: np.sqrt(np.mean((p-d.lb)**2))
v9b=lambda d:0.5*(d.lt1+d.lt2)/2+0.25*(d.chr_e3+d.wx_chr_st)
for fo in folds:
    d=D[fo].copy()
    for m in ('lgbL','mlp'):
        try: x=pd.read_csv(f'results_v10/cv_{m}_{fo}.csv'); d=d.merge(x[K+['pred']].rename(columns={'pred':m}),on=K)
        except FileNotFoundError: pass
    d['date']=pd.to_datetime(d.date)
    out=[f'{fo} v9b {r(d,v9b(d)):.4f}']
    for m in ('lgbL','mlp'):
        if m in d: out+= [f'{m} alone {r(d,d[m]):.4f}']+[f'+{m}×{w} {r(d,(1-w)*v9b(d)+w*d[m]):.4f}' for w in (0.05,0.1,0.15,0.2,0.3)]
    if 'mlp' in d and 'lgbL' in d: out+=[f'+both x.1 {r(d,0.8*v9b(d)+0.1*d.mlp+0.1*d.lgbL):.4f}']
    print(' | '.join(out))
