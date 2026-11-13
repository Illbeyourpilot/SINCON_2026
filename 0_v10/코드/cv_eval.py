import pandas as pd, numpy as np
from scipy.optimize import nnls
D=pd.read_pickle('cv_all.pkl')
M=['chr_daily','chr_e3','lt1','lt1n','lt2','struct','tfm','v3_x10','v3_x8','wx_chr','wx_chr_cl','wx_chr_clh','wx_chr_ft','wx_chr_st']
def tree(d): return 0.75*(d.lt1+d.lt2)/2+0.25*(d.v3_x10+d.v3_x8)/2
def chron(d): return (d.chr_e3+d.wx_chr)/2
def v8d(d): return 0.5*tree(d)+0.5*chron(d)
r=lambda d,p: float(np.sqrt(np.mean((p-d.lb)**2)))
rows=[]
def add(name,f):
    rows.append((name,r(D['A'],f(D['A'])),r(D['B'],f(D['B']))))
add('v8d',v8d)
for m in M: add('alone '+m,lambda d,m=m:d[m])
for t in (0.4,0.6,0.7): add(f'tree {t}/Chronos {1-t:.1f}',lambda d,t=t:t*tree(d)+(1-t)*chron(d))
for m in M:
    for w in (0.1,0.2,0.3):
        add(f'v8d×{1-w:.1f} + {m}×{w}',lambda d,m=m,w=w:(1-w)*v8d(d)+w*d[m])
CH=['chr_e3','wx_chr','wx_chr_cl','wx_chr_clh','wx_chr_ft','wx_chr_st','chr_daily','tfm']
import itertools
for a,b in itertools.combinations(CH,2):
    add(f'Chronos part=({a}+{b})/2',lambda d,a=a,b=b:0.5*tree(d)+0.25*(d[a]+d[b]))
for c in CH: add(f'Chronos part={c}',lambda d,c=c:0.5*tree(d)+0.5*d[c])
add('tree part + lt1n (lt1+lt2+lt1n)/3',lambda d:0.5*(0.75*(d.lt1+d.lt2+d.lt1n)/3+0.25*(d.v3_x10+d.v3_x8)/2)+0.5*chron(d))
add('tree part LGB only',lambda d:0.5*(d.lt1+d.lt2)/2+0.5*chron(d))
R=pd.DataFrame(rows,columns=['candidate','A','B'])
b=R.iloc[0]
R['dA']=R.A-b.A; R['dB']=R.B-b.B
both=R[(R.dA<0)&(R.dB<0)].sort_values('dB')
print('v8d', round(b.A,4), round(b.B,4))
print('\n== single models\n',R[R.candidate.str.startswith('alone')].round(4).to_string(index=False))
print('\n== improves on both folds (top 25 by dB)\n',both.head(25).round(4).to_string(index=False))
print('\nimproves on both folds:',len(both),'/',len(R))
# cross-fit NNLS: weights fit on A -> scored on B, and vice versa
for fit,ev in (('A','B'),('B','A')):
    X=D[fit][M].values; w,_=nnls(X,D[fit].lb.values); w=w/w.sum()
    print(f'NNLS fit {fit} -> score {ev} {r(D[ev],D[ev][M].values@w):.4f} (v8d {r(D[ev],v8d(D[ev])):.4f})', dict(zip(M,np.round(w,2))))
R.to_csv('cv_candidates.csv',index=False,encoding='utf-8-sig')
