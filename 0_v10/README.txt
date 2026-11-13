SINCON 2026 최종 v10 (2026-11-13 정리)
=========================================

[제출할 파일]
제출파일/submission_h9_v10.csv   <- 1순위. 팀 최고 h9(공개 0.153022) + (v10 - v8d) 로그 차이
제출파일/submission_v10.csv      <- 2순위. v10 단독 (h9 없이)
(검증: 157,194행, 결측 0, 음수 0, 열 example_id,boardings, example_id 순서 = test.csv)

[v10 구성 (log1p 공간)]
v10 = 0.8 × [ 0.5 × LightGBM(lt1·lt2 평균) + 0.5 × Chronos-2(chr_e3 + wx_chr_st 평균) ]
    + 0.2 × 가벼운 LightGBM(lgbL, seed 3개 평균)
    -> v8d 후보정 그대로(날씨·경기·명절·공연·신규역)
v8d 대비: XGBoost 제외, Chronos wx_chr -> wx_chr_st(구조 공변량), lgbL 추가

[검증 점수 RMSLE, 보정 전]   A=2025-07~10 학습<=6/30 / B=2026-03~06 학습<=2/28
v8d   A 0.2627  B 0.1831
v9b   A 0.2609  B 0.1809
v10   A 0.2559  B 0.1795   (8개월 모두 개선, seed 3개 안정)
미채택: MLP(A만 개선), lt1 비중 조정·struct 등(개선 0.001 미만)

[공개 점수 추정 (제출 없이) - 코드/estimate_public.py]
알려진 점수 2개(v8d 0.15475, h9 0.153022)와 검증 구간의 오차-변화 관계로 추정
  h9_v10  약 0.1492 (범위 0.1482~0.1502)
  v10     약 0.1493 (범위 0.1483~0.1503)
  h9_v9b  약 0.1507 (범위 0.1499~0.1516)
  v9b     약 0.1526 (범위 0.1518~0.1535)
주의: public 구간이 테스트 전체와 같다고 가정한 추정치. 실제 점수와 다를 수 있음

[폴더]
제출파일/    최종 제출 후보 2개
입력/        sub_v8dh9.csv(팀 h9), submission_v8d.csv(v8d = 루트 submission.csv 사본),
             calendar.csv, munhak_schedule_2026H2.csv, external_2026_07_10.csv
코드/        exp3_newmodels.py (lgbL·MLP 학습, A/B/full)
             final_post_v9.py  (혼합·보정. 환경변수 FP_XGB_W, FP_CHR, FP_EXTRA)
             make_v10_kernel.py (캐글 노트북 생성)
             cv_eval*.py       (혼합 후보 검증)
모델예측/    testpred_*.csv (테스트 기간 각 모델 로그 예측: lt1, lt2, chr_e3, wx_chr_st, lgbL)
캐글노트북/  nb.ipynb + kernel-metadata.json  (Kaggle: doooyeon/sincon-v10-comp)

[캐글 노트북]
- 대회 sincon-2026 데이터만 연결 (개인 데이터셋 연결 안 함). GPU T4, 인터넷 On
- 대회 데이터에 없는 달력·주최 측 날씨(날짜 단위)·h9−v8d 로그 차이는 노트북 안에 포함
  (노트북 1MB 제한 때문에 h9−v8d 차이는 0.001 단위로 저장, 로그 오차 최대 0.0005)
- 출력: submission.csv (= h9_v10), submission_v10.csv, submission_h9_v10.csv

[재현 (로컬)]
1. SINCON_v9_실험/v9a_work 에 testpred_*.csv 가 있는 상태에서
   FP_XGB_W=0 FP_CHR=chr_e3_full,wx_chr_st_full FP_EXTRA=lgbL_full:0.2 FP_OUT=submission_v10.csv python final_post_v9.py .
2. h9_v10 = expm1( log1p(h9) + log1p(v10) − log1p(v8d) ), 0 이상

[다른 폴더]
이전_제출후보/  submission_v9a, v9b, h9_v9b (이전 단계 후보)
실험기록/       exp2_newfeat.py, exp2_B.log (새 달력 변수 검증, 미채택)
SINCON_v9_실험/ 전체 실험 작업 폴더 (검증 예측 results_dl, results_v10 등)
