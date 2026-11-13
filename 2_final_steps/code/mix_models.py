"""재학습 모델 테스트 예측(log1p) 평균을 w 비율로 혼합. 행사 보정이 들어간 행(data/evmask.csv)은 제외.
사용: python mix_models.py in.csv out.csv w evmask.csv pred1.csv [pred2.csv ...]"""
import sys, numpy as np, pandas as pd
inp, out, w, evm = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
s = pd.read_csv(inp)
X = np.mean([pd.read_csv(p).set_index("example_id").pred_log.reindex(s.example_id).to_numpy() for p in sys.argv[5:]], axis=0)
em = pd.read_csv(evm).set_index("example_id").ev.reindex(s.example_id).fillna(0).to_numpy().astype(bool)
P = np.log1p(s.boardings.to_numpy()); ok = ~np.isnan(X) & ~em
Q = np.where(ok, (1 - w) * P + w * X, P)
pd.DataFrame({"example_id": s.example_id, "boardings": np.expm1(np.clip(Q, 0, None))}).to_csv(out, index=False)
print("mixed rows", ok.sum(), "w", w, "models", sys.argv[5:])
