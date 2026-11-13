# SINCON 2026 — 최종 예측 재현 자료 (팀명: ____ )

> 이 폴더는 **팀원 파이프라인 산출물(`data/input_sub_Gsh_prehol.csv`) → 최종 예측**까지의 단계를 담습니다.
> 입력 파일을 만드는 앞단(v8dh17, v10 학습, 블렌드, 팀원 보정)은 팀원 패키지에 있으며, 전체 순서는 아래 0절에 있습니다.

## 0. 전체 실행 순서 (제공 데이터 → 최종 예측)
| 순서 | 단계 | 담당/위치 | 산출물 |
|---|---|---|---|
| 1 | v8d 기본 모델 (LGB 2, XGB 2, Chronos-2 2) + v8d 후처리 | 팀원 패키지 (v8d 재현 노트북) | submission_v8d |
| 2 | v10 (LGB lt1·lt2 + Chronos chr_e3·wx_chr_st + lgbL) | 팀원 패키지 (최종_v10) | submission_v10_noghost_nocanc |
| 3 | v8dh17 (2024 하반기 합성 자료로 LGB/XGB/Chronos 재학습, seed 평균) | 팀원 패키지 | v8dh17 예측 |
| 4 | 블렌드 0.6×v8dh17쪽 + 0.4×v10 (log1p) | 팀원 패키지 | — |
| 5 | 행사·추세 보정 12개 (weekC 1.0배, 나머지 0.65배) | 코드: `records/prev_chain_code`, 적용·계수: 팀원 패키지 | — |
| 6 | 팀원 보정: 전체 −1%, 7월 +3%, 9월 −2%, 22~23시 −3%, 10~16시 +1%, 폭염(30℃ 초과 1℃당 −0.57%), 9/23 추석 귀성일 | 팀원 패키지 | `data/input_sub_Gsh_prehol.csv` (공개 0.146211) |
| 7 | **PLAVE 공연 보정** | 이 폴더 `code/post_concert.py` | output/step1_plave.csv |
| 8 | **작년 같은 시기 혼합** | 이 폴더 `code/post_lyana2.py` | output/step2_ly.csv |
| 9 | **인접역 피처 LightGBM 재학습(3 seed) 10% 혼합** | 이 폴더 `code/train_lgb.py`, `code/mix_models.py` | `predictions/final.csv` |

## 1. 실행 환경 및 방법
| 단계 | 내용 |
|---|---|
| 환경 준비 | Windows 11 Home 10.0.26200 / Git Bash, Python 3.13.2, CPU 8코어(GPU 불필요). `pip install -r requirements.txt` |
| 학습 | `bash run.sh retrain` → `code/train_lgb.py configs/lgb_nbr.json` (sincon_v2.py full 모드, LightGBM 3개, 각 약 10분). 초기 모델 없음(처음부터 학습). 검증 분할 없음(전체 학습, 기준일 2026-06-30) |
| 최종 추론 | `bash run.sh` (저장된 `models/testpred_*.csv` 사용, 약 1분) → `output/final.csv`. 마지막 로그(`logs/step4_check.log`)에 제출본과의 최대 차이 출력 |
| 재현 변동 | 7·8단계: 결정적(무작위 없음), 반복 결과 차이 0.0. 9단계 LightGBM: seed 0, 2, 3 고정이나 멀티스레드 부동소수 합산 순서 때문에 재학습 시 소수 미세 차이 가능(`bash run.sh` 은 저장 예측 사용 → 차이 0.0) |

## 2. 단계 설명과 계수 결정 방법
### 7. PLAVE 공연 보정 (`post_concert.py`, w=1.0)
- 대상: 2026-09-12·13 문학 주경기장 PLAVE 콘서트(19시, 공표 일정), 문학경기장 반경 내 역 / 360행
- 방법: 22시 정점형 저녁 공연일(박효신 3회 + 2025-07-26)의 실제 승차로 역×시간 편차를 재고, 예측 = 평소 주말 예측 + w×편차
- 결정: 과거 저녁 공연 4회를 하나씩 빼고 맞히는 LOO 검증으로 w 결정. 입력 파일 22시 예측 150·85명 → 7,630명
### 8. 작년 같은 시기 혼합 (`post_lyana2.py`)
- A = 2025년 같은 요일(350·357·364·371·378일 전, 공휴일 lag 제외) log1p 승차 중앙값 + 전체 평일 성장률(2026-05~06 / 2025-05~06). 2025-06-28 신규역 개통으로 수준이 바뀐 역은 개통 전후 변화만큼 보정
- 최종 = (1−w)·입력 + w·A, 평일 w=0.10, 주말 w=0.05. 제외: 신규역 3107~3109, 공휴일·추석 연휴 전후, 문학·부천종합운동장·아시아드 주변역
- 결정: 방법은 폴드 B(2026-02-28 기준 3~6월) 검증에서 개선(−0.002) 확인. **계수(w, lag 수, 성장률 방식)는 연습 서버(sincon, 공개 구간 동일) 공개 점수를 보고 선택** — 기록 `records/practice_submissions_sincon.csv`
### 9. 인접역 피처 LightGBM 재학습 + 혼합
- 설정 `configs/lgb_nbr.json`: v8d lt1 설정(num_leaves 255, min_data_in_leaf 100, feature_fraction 0.6, lr 0.03, 2500 rounds)에 인접역 피처 그룹 `nbr`(노선+1.5km 인접 역 수준·추세) 추가, seed 0·2·3
- 혼합: 세 seed 평균을 10% (행사 보정 행 `data/evmask.csv` 제외)
- 결정: **혼합 비율 10%는 연습 서버 공개 점수로 선택**(5%·10%·15%·20% 비교)

## 3. 데이터 작성표
| 파일 | 구분 | 내용 | 출처/생성 |
|---|---|---|---|
| data/train.csv, test.csv, metaData.csv | 제공 | 변형 없음 | 대회 |
| data/weather_incheon_hourly.csv | 외부 | 인천 시간별 기온·강수·적설·풍속(2024-12~2026-10) | Open-Meteo Historical Weather API |
| data/ssg_home_schedule.csv, ssg_home_actual.csv | 외부 | SSG 문학 홈경기 날짜·시작시각(원 일정/실제 개최) | KBO 공식 일정 |
| data/bucheon_home_schedule.csv, bucheon_home_actual.csv | 외부 | 부천FC 홈경기 날짜·시작시각 | K리그 공식 일정 |
| data/station_geo_graph.csv | 파생 | 역 좌표, 노선 그래프 특징, 주요 시설까지 거리 | 공개 지도(OSM 등) 좌표로 계산 (v8d aux_data) |
| data/evmask.csv | 파생 | 행사 보정이 들어간 행 표시 | `provenance/make_evmask.py` (보정 체인 중간 파일 비교) |
| data/input_sub_Gsh_prehol.csv | 파생 | 팀원 파이프라인 산출물 | 팀원 패키지 (0절 1~6) |
| models/testpred_lt1nbr*_full.csv | 산출물 | 재학습 LightGBM 테스트 예측(log1p) | `code/train_lgb.py` |
| 코드 안 날짜 | 외부 | PLAVE·과거 공연 일정 | 주최 측·예매처 공지 |

**사용하지 않은 것(해당 없음):** 평가 기간(2026-07~10) 승하차 정답 및 관련 자료, 하차(alightings, 이 폴더 단계), 사전학습 가중치(이 폴더 단계), STCIS 2024 자료(실험만 하고 최종 미사용).
**평가 정답 열람·확보·사용 사실:** 없음 (이 폴더 단계 기준).

## 4. 모델 및 중간 산출물
- 재학습 LightGBM: 가중치 파일은 저장하지 않음, 테스트 예측(`models/`)과 재생성 방법(`bash run.sh retrain`) 제공
- 학습 로그: `logs/log_lt1nbr*_full.txt`, `logs/train_*_stdout.log`

## 5. 기록 (records/)
- `practice_submissions_sincon.csv`, `submissions_sincon-2026.csv`: 연습·본 대회 제출 기록(설명란에 각 시도 내용)
- `prev_chain_*`: 0절 5단계 보정 12개 코드·설명
- `experiments/`: 검증 실험 코드(채택·미채택 포함: 공휴일 보정, 날씨 지연, 분해 모델, 작년 대비 변화 모델, 블렌드 후보 CV 등)

## 6. 자체 점검
- [x] 최종 예측 파일과 코드·모델·설정의 관계 명시 (0절, 2절)
- [x] 데이터·모델 작성표, 미사용 항목 표시 (3·4절)
- [x] 경로·명령·의존성·시드 (1절) — 앞단은 팀원 패키지
- [x] 평가 정답 관련 자료 사용 사실 없음 (3절) — 팀원 단계(합성 자료 등)는 팀원 패키지에서 기재 필요
