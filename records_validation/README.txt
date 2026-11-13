검증 예측 (폴드 A: 2025-06-30까지 학습 → 2025-07~10 예측 / 폴드 B: 2026-02-28까지 학습 → 2026-03~06 예측)
- cv_all.pkl : dict{A,B} -> DataFrame(date, station_id, hour, lb, 모델별 예측 log1p: lt1, lt2, v3_x10, v3_x8, chr_e3, wx_chr, wx_chr_st, tfm 등)
- cv_folds_v8dh13_v8dh17/ : v8dh13, v8dh17 폴드 예측 (y_log1p, pred_log1p, pred)
- results_v10/ : v10 의 가벼운 LightGBM(lgbL), MLP 폴드 예측
