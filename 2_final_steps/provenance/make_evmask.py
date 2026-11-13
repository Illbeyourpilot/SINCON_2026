"""data/evmask.csv 생성 기록: 우리 행사 보정 체인(records/prev_chain_code)이 값을 바꾼 행 = 1.
체인 중간 파일(cand/*.csv)을 비교해 만든 것. 재현: records/prev_chain_run.sh 를 팀원 블렌드 입력으로 돌린 뒤 아래 실행."""
import pandas as pd, numpy as np
c="cand/"
pairs=[("tv9/v9_A_C100.csv","T0_ACD.csv"),("T0_ACD.csv","U1.csv"),("U3.csv","U4.csv"),("U4.csv","U5.csv"),("U5.csv","U6.csv"),
       ("U6.csv","U7.csv"),("U7.csv","U8.csv"),("U8.csv","U9.csv"),("U9.csv","U10.csv"),("U10.csv","../blend60_ourfix_safe2.csv")]
ev=None
for a,b in pairs:
    x=np.log1p(pd.read_csv(c+a).set_index("example_id").boardings); y=np.log1p(pd.read_csv(c+b).set_index("example_id").boardings).reindex(x.index)
    m=(x-y).abs()>1e-6; ev=m if ev is None else (ev|m)
te=pd.read_csv("test.csv",parse_dates=["date"]).set_index("example_id")
ev=ev|(te.station_id.isin([3209,3210])&te.date.isin(pd.to_datetime(["2026-09-12","2026-09-13","2026-10-17","2026-10-18"]))).reindex(ev.index)
pd.DataFrame({"example_id":ev.index,"ev":ev.astype(int).values}).to_csv("evmask.csv",index=False)
