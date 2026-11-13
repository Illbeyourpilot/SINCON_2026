"""configs/lgb_nbr.json 의 각 설정으로 sincon_v2.py full 모드 학습 -> models/testpred_<tag>.csv
사용: (data/ 에서) python ../code/train_lgb.py ../configs/lgb_nbr.json ../models_retrained"""
import json, os, sys
cfgs = json.load(open(sys.argv[1], encoding="utf-8")); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
os.environ["SINCON_OUT"] = out
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sincon_v2.py"), encoding="utf-8").read()
for c in cfgs:
    g = {"__name__": "__main__", "CONFIG_OVERRIDE": c}
    exec(compile(src, "sincon_v2.py", "exec"), g)
