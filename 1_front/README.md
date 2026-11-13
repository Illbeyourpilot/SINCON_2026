# 앞단 재현 패키지 (우리 쪽: v8dh17 학습 → 앙상블·보정 → 팀원 블렌드 → 제출본)

## 결과물 (run_final.sh 로 재현, 제출본과 차이 0.0 확인)
| 파일 | 공개 점수 | 내용 |
|---|---|---|
| sub_Gsh_prehol.csv | 0.146211 | 팀원 쪽 출발점 (PLAVE·작년혼합·nbr 보정이 이 위에 얹힘) |
| sub_Gng_prehol.csv | - | 위 + 인접 그래프 피처 뺀 LightGBM 혼합 (CV B LGB 0.1633→0.1622) |
| sub_safe_Gng.csv | 0.147460 | 공개 점수로 정한 수준 조정 제외판 |
최종 혼합: sub_mix_tbest_ng = 팀원 best_Gsh_plave_ly_nbr10 + (log sub_Gng_prehol − log sub_Gsh_prehol) → 0.145565

## sub_BASE_noghost_nocanc.csv (팀원 블렌드 input 의 우리 쪽, run_final.sh 0단계)
`cd work && FP_LGB=lt1h_full,lt2h_full FP_XGB=xh10_full,xh8_full,xh10_s2,xh8_s3 FP_CHRHIST=0.25 FP_NEWTREND=0.5 FP_WT=0.55 FP_WL=0.6 FP_NN=nn13489_full:0.3 FP_SSGBOX=0.5 FP_BUS=0.75 FP_HOLMATCH=0 FP_GHOST_OFF=1 FP_CANCEL_FIX=1 FP_OUT=sub_BASE_noghost_nocanc.csv python ../src/final_post.py .`
(Chronos 는 기본값 chr_e3_full,wx_chr_full) — FP_GHOST_OFF=1: 고스티벌 10/24~25 취소(사후 가산 제거 + 문학 3127 인근 같은 요일 평균), FP_CANCEL_FIX=1: 우천취소 8/5·8/6·8/22 문학 경기일을 인근 같은 요일 비경기일 평균으로.
같은 명령에서 두 값을 0 으로 두면 sub_v8dh13.csv (= sub_BASE.csv, 공개 0.152756).

## 실행
1. (모델 학습, Kaggle) kernels/*/config_head.txt 의 CONFIG_OVERRIDE 한 줄 + src/sincon_v2.py 전체 = 커널 run.py.
   Chronos 커널은 pip 2줄 + CONFIG + sincon_v2.py, NN 커널은 `python src/make_nn_kernel.py <slug> '<json>'`.
   데이터셋: data/*.csv + ext/*.csv + ext/syn2024/*.csv. 출력 testpred_{tag}.csv (example_id, pred_log) → work/ 에 이미 넣어 둠. 로그: logs/
2. (합성 자료) `python src/syn2024.py make 2026-06-30 full` (A: 2025-06-30, B: 2026-02-28) → ext/syn_2024h2_{tag}.csv
3. `bash run_final.sh` → out_repro/ 에 제출본 3개

## 사용 모델 (test 예측 → work/)
- LightGBM 2024판: lt1y_full, lt1y_s2 (num_leaves 255), lt2y_full, lt2y_s3 (63), lt1yng_full (graph 그룹 제외)
- XGBoost 2024판: xh10y_full, xh10y_s2 (depth 10), xh8y_full, xh8y_s3 (depth 8)
- Chronos-2 2024판: chr_e3y_full, wx_chry_full (날씨 공변량)
- 신경망: nn13b18b9_full = nn1, nn3, nn8, nnb1, nnb9 평균
- 신규역 3107~3109 는 2024 이전판 (lt1h/lt2h_fullb, xh10/xh8_fullb·s2b·s3b, chr_e3_full, wx_chr_full)

## 2024 하반기 합성 자료 (src/syn2024.py) — 평가 기간 정보 미사용
- 일별 역 합계: data.go.kr 15004329 구버전 실측 2024-07~2025-06 (ext/daily_hist)
- 시간대 비율 LightGBM: 각 기준일(A 2025-06-30 / B 2026-02-28 / full 2026-06-30) 까지 train 만 학습
- 피처: ICTR 2024 월별 시간대 비율(2025 이전 자료), 2024 날씨(open-meteo), 2024 SSG 경기, 요일·휴일
- 2026-07~10 자료는 어디에도 쓰지 않음

## 후처리 단계
1. src/final_post.py: 앙상블(트리 0.65 : Chronos 0.35, NN 0.3) + 보정 12단계. 설정은 run_final.sh 의 FP_* 값
2. src/lb_chain.py: 팀원 블렌드(우리 0.6 / v10 0.4, teammate/input), 팀원 보정 (teammate/output 과의 차: weekC 1.0배 재계산, 나머지 11개 0.65배),
   수준 조정 전체 −1%, 7월 +3%, 9월 −2%, 22~23시 −3%, 10~16시 +1% (LC_* 로 끔)
3. src/heat_post.py: 최고기온 30도 초과 1도당 −0.57% (CV A 회귀, 서울교통공사 보고와 일치)
4. src/prehol_post.py: 추석 직전 평일 9/23 시간대 패턴 (과거 연휴 직전 평일 3회 평균) x0.7

## 계수 결정 근거 (설명서 기재용)
- CV(A·B) 로 정함: 2024 합성 사용, 트리 0.65, 폭염 0.57%, graph 제외 LGB 혼합, 신경망 비중
- 공개 점수로 정함 (연습 서버): 팀원 보정 0.65배, 수준 조정 5개, FP_CHRHIST/SSGBOX/BUS/WL/NEWTREND, 블렌드 0.6
