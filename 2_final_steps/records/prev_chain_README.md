# 재현: blend60_ourfix_safe8.csv (Public LB 연습 0.14918)

## 실행
```
bash run.sh
```
결과: `output/blend60_ourfix_safe8.csv` (제출본과 완전히 동일, 최대 차이 0.0 확인)
소요 약 2~3분, GPU 불필요. Python 3 + pandas, numpy, lightgbm, pyarrow.

## 구조
| 폴더 | 내용 |
|---|---|
| input/ | 팀원 블렌드 `submission_blend_base60_v10_40.csv` (0.6·sub_BASE_noghost_nocanc + 0.4·v10_noghost_nocanc, LB 0.151116) |
| code/ | 보정 스크립트 16개 + 공용 모듈(sincon_v2, final_post, stack_cv, cv_aci, cv_bucheon) |
| data/ | train/test, 날씨, SSG·부천FC 일정·관중, 컨벤시아 일정(conv26), 역 좌표, v8d 개별 모델 테스트 예측(testpred_*, 해변축제 보정 제거용) |
| output/ | 결과 + 중간 파일(tmp/) |

## 적용 보정 (순서대로, log 공간)
0a. 9/28 문학 주변 추석 보정에 섞인 2025 준PO 급증 제거 (post_v9 A)
0b. 9~10월 일반 평일 수준을 상반기 전년비(1.032)로 맞춤 (post_v9 C)
0c. PLAVE 9/12~13 문학 저녁 공연 22시 급증 (post_concert D, 박효신 3회+2025-07-26 편차)
1. 펜타포트 7/31~8/2 — 2025 편차 (3137~3139)
2. 7~8월 평일 수준 — 주간 YoY, 목표비 2026/2025 5~6월 = 1.0295
3. 아시아드 3209/3210 야간 이상치 정리 (9/12·13, 10/17·18)
4. 신규역 3107~3109 감쇠 추세 phi=0.8 (목표 수준으로 맞춤 → 팀원 추세와 중복 없음)
5. PSY 7/11 아시아드 (2025-06-28 편차)
6. 송도컨벤시아 전시 3136, k=0.4
7. 아트센터인천·트라이보울 저녁 공연 3136/3137, k=0.6
8. 부천FC K1 홈경기 3753~3756, 관중수 log 스케일
9. 가을 연휴 인천대공원 3226·운연 3227
10. v8d 송도해변축제 보정 제거(8/8, 8/15) + 능허대 축제 선학 3128
11. 추석 당일 9/25 부평삼거리 3122 (가족공원 성묘)
12. 송도맥주축제 8/22~30 야간 3138/3139 재보정, ratio 0.45
13. 주말(비공휴일) 전년 대비 수준: 2026/2025 5~6월 주말 비율(1.038)로 ±45일 창, k=0.5 (fold B 0.17391→0.17320)
14. 작년 같은 시기 모델 혼합 w=0.15: 2025 같은 요일(364±7일) 실제 log 중앙값 + 역별 성장률(5~6월 YoY, 개통 전후 수준 변화 보정). 공휴일·명절 전후·경기장 주변역·신규역 제외 (fold B -0.002)

제외: 우천취소 경기 보정(입력 파일에 이미 반영), Sustainable Wave·이슬라이브·가재울 둘째 토요일(검증 약함, full판 0.150426로 확인).

## 데이터 사용 원칙
평가 기간(2026-07~10) 실제 승하차는 사용하지 않음. 모든 편차는 2025-01~2026-06 학습 데이터에서 계산.
외부 데이터: 행사 일정·관중수·날씨 (대회 규정상 허용).

## 입력 파일 재현
input/ 파일은 팀원 파이프라인 산출물. 처음부터 재현하려면 팀원의 sub_BASE(v8dh13)·v10 재현 코드 필요.

## 다른 제출 파일 위에 이 보정 얹기
```
bash run.sh <입력 제출.csv 절대경로> <출력.csv 절대경로>
# 예: bash run.sh /path/sub_cmb_Gs.csv /path/sub_cmb_Gs_ourfix.csv
```
- 입력은 example_id,boardings 형식의 제출 파일. 결과는 그 위에 보정 16단계를 차례로 적용한 파일.
- 입력에 이미 같은 보정이 들어 있으면 run.sh 에서 해당 줄을 # 으로 주석 처리하고 돌릴 것.
  - 대부분 "기준선 + 행사 편차"로 다시 계산하는 방식이라 겹쳐도 두 번 더해지지는 않음.
  - 곱셈형이라 겹치면 두 번 들어가는 것: post_park(대공원·운연), post_beach_nhd(능허대 선학 +10%), post_v9 C(9~10월 평일 수준), post_weekend, post_lyana(작년 같은 시기 혼합).
- 고스티벌 취소·우천취소 경기 처리는 이 체인에 없음(입력 쪽에서 처리돼 있다고 가정).

## 사용 데이터 전체 목록 (검증 제출용)
### 대회 제공
- data/train.csv (2025-01-01~2026-06-30 승하차, 학습에 쓴 승차 정답은 이것뿐), data/test.csv, data/metaData.csv

### 외부 데이터 (전부 메타데이터, 2026-07~10 승하차 정답 미사용)
| 파일 | 내용 | 출처 |
|---|---|---|
| data/weather_incheon_hourly.csv | 인천 시간별 기온·강수·적설·풍속 2024-12~2026-10 | Open-Meteo Historical Weather API (공개) |
| data/ssg_home_schedule.csv, ssg_home_actual.csv | SSG 문학 홈경기 날짜·시작시각(원 일정/실제 개최) | KBO 공식 일정 |
| data/bucheon_home_schedule.csv, bucheon_home_actual.csv | 부천FC 홈경기 날짜·시작시각 | K리그 공식 일정 |
| data/bucheon_att.csv | 부천FC 경기장 관중 수(2026) — 지하철 승차 아님 | K리그 데이터포털 경기 기록 |
| data/conv26.csv | 송도컨벤시아 2026 행사명·기간·종류 | 송도컨벤시아 행사 일정(공개) |
| data/station_geo_graph.csv | 역 좌표·노선 그래프 특징 | OSM 등 공개 지도 |
| code/post_*.py 안의 날짜 | 펜타포트·PSY·Sustainable Wave·PLAVE·이슬라이브·아트센터 공연·맥주축제·능허대 축제·고스티벌 취소 등 행사 일정 | 주최 측·예매처 공지, 보도(각 스크립트 docstring) |

### 입력 파일(input/)의 출처
- input/submission_blend_base60_v10_40.csv = 0.6×sub_BASE_noghost_nocanc + 0.4×submission_v10_noghost_nocanc (log1p 공간)
  - sub_BASE(v8dh13), v10 은 팀원 파이프라인 산출물 → 각 재현 패키지(코드·입력·출력) 별도 첨부 필요
  - v8d 원본 재현: C:/sincon/v8d/proj (sincon_final.ipynb, src/, aux_data/)

### 사용하지 않은 것
- 2026-07~10 승하차(대회 정답), 해당 기간 역별 승하차 뉴스, 하차(alightings) 컬럼(보정 코드에서 미사용)
- STCIS 2024 자료(실험만 하고 최종 파일에는 미사용)
