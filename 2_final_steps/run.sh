#!/bin/bash
# 사용: bash run.sh            -> 저장된 재학습 예측(models/) 사용, 1분
#       bash run.sh retrain    -> LightGBM 2개 재학습(CPU 약 20분) 후 그 예측 사용
set -e
cd "$(dirname "$0")/data"; mkdir -p ../output ../logs
M=../models
if [ "$1" == "retrain" ]; then python ../code/train_lgb.py ../configs/lgb_nbr.json ../models_retrained 2>&1 | tee ../logs/retrain_run.log; M=../models_retrained; fi
python ../code/post_concert.py train.csv test.csv . ../code/sincon_v2.py input_sub_Gsh_prehol.csv ../output/step1_plave.csv 1.0 2>&1 | tee ../logs/step1_post_concert.log
python ../code/post_lyana2.py ../output/step1_plave.csv ../output/step2_ly.csv 0.10 0.05 350,357,364,371,378 0.0 st 2>&1 | tee ../logs/step2_post_lyana2.log
python ../code/mix_models.py ../output/step2_ly.csv ../output/final.csv 0.10 evmask.csv $M/testpred_lt1nbr_full.csv $M/testpred_lt1nbr_s2_full.csv $M/testpred_lt1nbr_s3_full.csv 2>&1 | tee ../logs/step3_mix.log
python -c "import pandas as pd,numpy as np;a=pd.read_csv('../output/final.csv');b=pd.read_csv('../predictions/final.csv');print('rows',len(a),'max diff vs submitted',float(np.abs(a.boardings-b.boardings).max()))" | tee ../logs/step4_check.log
