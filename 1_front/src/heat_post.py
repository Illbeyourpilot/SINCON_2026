"""폭염 보정: 최고기온 30도 초과 1도당 log 예측 -HEAT_S (기본 0.0057, 폴드 A 회귀). 사용: python heat_post.py in.csv out.csv"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
S = float(os.environ.get("HEAT_S", 0.0057))
w = pd.read_csv(ROOT / "ext/syn2024/weather_incheon_hourly24.csv", parse_dates=["date"]); tm = w.groupby("date").temperature_2m.max()
te = pd.read_csv(ROOT / "data/test.csv", parse_dates=["date"]); b = pd.read_csv(sys.argv[1]); assert (b.example_id == te.example_id).all()
h = np.clip(te.date.map(tm) - 30, 0, None).to_numpy()
b.assign(boardings=np.expm1(np.clip(np.log1p(b.boardings) - S * h, 0, None))).to_csv(sys.argv[2], index=False); print("저장", sys.argv[2])
