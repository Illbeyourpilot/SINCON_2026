"""Build the v10 Kaggle notebook (only the sincon-2026 competition attached; aux data embedded).
LightGBM lt1, lt2 + Chronos chr_e3, wx_chr_st + light LightGBM (lgbL, 3 seeds)
-> final_post_v9 (XGB 0, lgbL 0.2) -> submission_v10.csv -> h9 + (v10 - v8d) = submission_h9_v10.csv (= submission.csv)"""
import base64
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
nb0 = json.loads((HERE / "sincon_final.ipynb").read_text(encoding="utf-8"))
recipe = json.loads((HERE / "src" / "final_recipe.json").read_text(encoding="utf-8"))
by = {s["tag"]: s for s in recipe["steps"]}
steps = [by["lt1_full"], by["lt2_full"], by["chr_e3_full"],
         dict(mode="chronos_full", tag="wx_chr_st_full", chr_struct={"K": 10},
              chr_geo=True, chr_bfc=True, chr_wx=True, ev_actual=True)]

# aux data not in the competition files -> embedded in the notebook
WXC = ["avg_temp", "max_temp", "min_temp", "rainfall", "avg_wind", "sunshine", "humidity", "has_rain"]
Wd = pd.concat([pd.read_csv(ROOT / f"weather_data/{f}_weather.csv", encoding="utf-8-sig") for f in ("train", "test")])
EMB_WX = Wd.groupby("date")[WXC].mean().reset_index().to_csv(index=False)  # values are identical within a date (checked) -> one row per date
EMB_CAL = (ROOT / "calendar.csv").read_text(encoding="utf-8-sig")
h9 = pd.read_csv(ROOT / "sub_v8dh9.csv")
v8 = pd.read_csv(ROOT / "submission.csv")
assert (h9.example_id.values == v8.example_id.values).all()
import lzma
D = np.log1p(h9.boardings.values) - np.log1p(v8.boardings.values)
EMB_D = base64.b64encode(lzma.compress(np.round(D * 1e3).astype(np.int16).tobytes(), preset=9 | lzma.PRESET_EXTREME)).decode()  # quantized to 1e-3 (log error <= 5e-4) to fit the ~1MB notebook limit

nb = copy.deepcopy(nb0)
c3 = "".join(nb["cells"][3]["source"])
i0 = c3.index("CODES = ") + len("CODES = ")
i1 = c3.index("\nfor name, text in CODES.items()")
codes = json.loads(c3[i0:i1])
codes["final_post.py"] = (HERE / "v9a_work" / "final_post_v9.py").read_text(encoding="utf-8")
codes["exp3_newmodels.py"] = (HERE / "exp3_newmodels.py").read_text(encoding="utf-8")
nb["cells"][3]["source"] = [c3[:i0] + json.dumps(codes, ensure_ascii=False) + c3[i1:]]
c4 = "".join(nb["cells"][4]["source"])
nb["cells"][4]["source"] = ["RECIPE = " + repr({"steps": steps}) + "\n" + c4[c4.index("\ndef run("):]]

cell5 = (
    "# aux data (not in competition files): calendar, organizer daily weather, log(h9) - log(v8d)\n"
    f"EMB_CAL = {EMB_CAL!r}\nEMB_WX = {EMB_WX!r}\nEMB_D = {EMB_D!r}\n"
    + """
import numpy as np, gzip, base64, io
(WORK / 'sincon-2026').mkdir(exist_ok=True); (WORK / 'weather_data').mkdir(exist_ok=True); (WORK / 'v10').mkdir(exist_ok=True)
for n in ('train.csv', 'test.csv'):
    shutil.copy(WORK / n, WORK / 'sincon-2026' / n)
(WORK / 'calendar.csv').write_text(EMB_CAL, encoding='utf-8-sig')
wx = pd.read_csv(io.StringIO(EMB_WX))
wx.to_csv(WORK / 'weather_data' / 'train_weather.csv', index=False)
wx.iloc[:0].to_csv(WORK / 'weather_data' / 'test_weather.csv', index=False)
(WORK / 'v10' / 'exp3_newmodels.py').write_text(CODES['exp3_newmodels.py'], encoding='utf-8')
run([sys.executable, '-u', str(WORK / 'v10' / 'exp3_newmodels.py'), 'full', '0', '1', '2'], env={'MODELS': 'lgbL'}, name='lgbL')
shutil.copy(WORK / 'v10' / 'results_v10' / 'testpred_lgbL_full.csv', WORK / 'testpred_lgbL_full.csv')

# blend + corrections -> v10
run([sys.executable, '-u', str(WORK / 'final_post.py'), str(WORK)],
    env={'FP_LGB': 'lt1_full,lt2_full', 'FP_XGB_W': '0', 'FP_CHR': 'chr_e3_full,wx_chr_st_full',
         'FP_EXTRA': 'lgbL_full:0.2', 'FP_OUT': 'submission_v10.csv'}, name='v10')

# h9 + (v10 - v8d) = v10 + (h9 - v8d)  (log space)
te = pd.read_csv(WORK / 'test.csv')
v10 = pd.read_csv(WORK / 'submission_v10.csv')
import lzma
D = np.frombuffer(lzma.decompress(base64.b64decode(EMB_D)), dtype=np.int16).astype(np.float64) / 1e3  # log(h9)-log(v8d)
assert len(D) == len(te) and (v10.example_id.values == te.example_id.values).all()
P = np.log1p(v10.boardings.to_numpy()) + D
pd.DataFrame({'example_id': te.example_id, 'boardings': np.expm1(np.clip(P, 0, None))}).to_csv(WORK / 'submission_h9_v10.csv', index=False)
for out_name in ('submission_v10.csv', 'submission_h9_v10.csv'):
    s = pd.read_csv(WORK / out_name)
    print(out_name, 'rows', len(s), 'na', int(s.isna().sum().sum()), 'neg', int((s.boardings < 0).sum()),
          'cols', list(s.columns), 'order_ok', bool((s.example_id.values == te.example_id.values).all()))
shutil.copy(WORK / 'submission_h9_v10.csv', WORK / 'submission.csv')
""")
nb["cells"][5]["source"] = [cell5]
nb["cells"][0]["source"] = [
    "# SINCON 2026 v10 최종 노트북\n",
    "입력: 대회 `sincon-2026` 데이터만 연결. 달력·주최 측 날씨(날짜 단위)·h9−v8d 차이는 노트북 안에 포함. GPU·인터넷 필요.\n",
    "v10 = 0.8·[LightGBM lt1·lt2 0.5 + Chronos-2 (chr_e3 + wx_chr_st) 0.5] + 0.2·가벼운 LightGBM(lgbL) → v8d 후보정.\n",
    "`submission.csv` = h9 + (v10 − v8d) (로그 공간).\n"]
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    else:
        c.pop("outputs", None)
OUT = HERE / "kaggle_kernels" / "v10_final"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "nb.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
meta = {"id": "doooyeon/sincon-v10-comp", "title": "sincon-v10-comp", "code_file": "nb.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": ["sincon-2026"],
        "kernel_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
print(json.dumps(meta))
