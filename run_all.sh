#!/bin/bash
# 전체 재현 (저장된 모델 테스트 예측 사용): 앞단 앙상블·보정 -> 마지막 3단계 -> predictions/final.csv 와 비교
# 모델 재학습까지 하려면 README 5.2 참고 (Kaggle 커널, LightGBM retrain)
set -e
R="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$R/logs"
echo "[1/4] 0단계 행사·추세 보정 12개 (blend60_ourfix_safe 재생성·검증)"
bash "$R/0b_our_corrections/run_safe.sh" "$R/1_front/teammate/input/submission_blend_base60_v10_40.csv" "$R/logs/blend60_ourfix_safe_repro.csv" > "$R/logs/0_run_safe.log" 2>&1
python -c "import pandas as pd,numpy as np,sys;a=pd.read_csv(sys.argv[1]);b=pd.read_csv(sys.argv[2]);print('  0단계 차이',float(np.abs(a.boardings-b.boardings).max()))" "$R/logs/blend60_ourfix_safe_repro.csv" "$R/1_front/teammate/output/blend60_ourfix_safe.csv" | tee -a "$R/logs/0_run_safe.log"
echo "[2/4] 앞단 앙상블·보정 (1_front/run_final.sh)"
bash "$R/1_front/run_final.sh" > "$R/logs/1_run_final.log" 2>&1
echo "[3/4] 마지막 3단계 (2_final_steps/run.sh)"
cp "$R/1_front/out_repro/sub_Gsh_prehol.csv" "$R/2_final_steps/data/input_sub_Gsh_prehol.csv"
bash "$R/2_final_steps/run.sh" > "$R/logs/2_run.log" 2>&1
echo "[4/4] 최종 확인"
python -c "import pandas as pd,numpy as np,sys;a=pd.read_csv(sys.argv[1]);b=pd.read_csv(sys.argv[2]);assert len(a)==157194 and a.boardings.notna().all() and (a.boardings>=0).all();print('  행',len(a),'| predictions/final.csv 와 최대 차이',float(np.abs(a.boardings-b.boardings).max()))" "$R/2_final_steps/output/final.csv" "$R/predictions/final.csv" | tee "$R/logs/3_check.log"
