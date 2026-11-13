import pandas as pd, numpy as np
D=pd.read_pickle('cv_all.pkl')
HSET=set(pd.to_datetime(["2025-01-01","2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-03-01","2025-03-03","2025-05-05","2025-05-06","2025-06-03","2025-06-06","2025-08-15","2025-10-03","2025-10-05","2025-10-06","2025-10-07","2025-10-08","2025-10-09","2025-12-25","2026-03-01","2026-03-02","2026-05-01","2026-05-05","2026-05-24","2026-05-25","2026-06-03","2026-06-06","2025-05-01"]))
for f,d in D.items():
    dt=pd.to_datetime(d.date); d['dtp']=np.where(dt.isin(HSET),2,np.where(dt.dt.dayofweek>=5,1,0)); d['mon']=dt.dt.month
r=lambda d,p: np.sqrt(np.mean((p-d.lb)**2))
L=lambda d:(d.lt1+d.lt2)/2
C=lambda d:(d.chr_e3+d.wx_chr_st)/2
v9b=lambda d:0.5*L(d)+0.5*C(d)
v8d=lambda d:0.5*(0.75*L(d)+0.25*(d.v3_x10+d.v3_x8)/2)+0.25*(d.chr_e3+d.wx_chr)
out=[]
def add(n,f): out.append((n,r(D['A'],f(D['A'])),r(D['B'],f(D['B']))))
add('v8d',v8d); add('v9b',v9b)
for w in (0.4,0.6): add(f'v9b tree {w}',lambda d,w=w:w*L(d)+(1-w)*C(d))
for m in ['lt1n','struct','wx_chr_ft','wx_chr','chr_daily','wx_chr_clh','tfm','v3_x8']:
    for w in (0.05,0.1,0.2): add(f'v9b×{1-w:.2f}+{m}×{w}',lambda d,m=m,w=w:(1-w)*v9b(d)+w*d[m])
add('tree = lt1 only',lambda d:0.5*d.lt1+0.5*C(d))
add('LGB lt1 0.7/lt2 0.3',lambda d:0.5*(0.7*d.lt1+0.3*d.lt2)+0.5*C(d))
add('Chronos e3+st+ft /3',lambda d:0.5*L(d)+0.5*(d.chr_e3+d.wx_chr_st+d.wx_chr_ft)/3)
add('Chronos e3+st+wx /3',lambda d:0.5*L(d)+0.5*(d.chr_e3+d.wx_chr_st+d.wx_chr)/3)
add('Chronos = st only',lambda d:0.5*L(d)+0.5*d.wx_chr_st)
# tree weight per day type: choose on one fold, score on the other
def fitw(d):
    W={}
    for t in (0,1,2):
        m=d.dtp==t; best=min(np.arange(0.2,0.81,0.05),key=lambda w:r(d[m],w*L(d[m])+(1-w)*C(d[m]))); W[t]=round(best,2)
    return W
for fa,fb in (('A','B'),('B','A')):
    W=fitw(D[fa]); d=D[fb]; w=d.dtp.map(W); print(f'day-type tree weight fit {fa} {W} -> {fb} {r(d,w*L(d)+(1-w)*C(d)):.4f} (v9b {r(d,v9b(d)):.4f})')
R=pd.DataFrame(out,columns=['candidate','A','B']); b=R[R.candidate=='v9b'].iloc[0]
R['dA']=R.A-b.A; R['dB']=R.B-b.B
print(R.round(4).to_string(index=False))
