# README

# SINCON 2026 — 인천 지하철 시간대별 승차 인원 예측 (팀 [컴온인지])

인천 지하철 71개 역의 2026-07-01 ~ 10-31 (123일), 06~24시 18개 시간
구간 승차 인원 157,194행을 예측한다. 평가 지표는 RMSLE (낮을수록 좋음).

- **최종 제출 파일**: `predictions/final.csv` (= `final_nbr10_3seed.csv`, Public 0.145536, sha256 앞자리 a1bc8abdbe7221db)
- **재현 명령**: `bash run_all.sh` 한 줄로 저장된 모델 예측에서 최종 파일까지 다시 만든다 (약 1분, 제출본과 최대 차이 7e-12). 모델 학습부터 다시 하는 방법은 5.2절.

---

## 1. 최종 모델 구성 한눈에 보기

```
 ┌────────────────────────────────────────┐     ┌──────────────────────────────┐
 │ 모델 A: v8dh17                          │     │ 모델 B: v10                   │
 │ LightGBM·XGBoost·Chronos-2·신경망 앙상블 │     │ LightGBM 2 + Chronos-2 2      │
 │ (2024 하반기 합성 자료 추가 학습)         │     │ + 가벼운 LightGBM             │
 └───────────────────┬────────────────────┘     └──────────────┬───────────────┘
                     └────────── 60 : 40 (log1p 공간) ──────────┘
                                         │
               행사·추세 보정 12개 (평일 수준 1.0배, 나머지 0.65배)
               수준 조정 · 폭염 · 추석 귀성일(9/23)
                                         │
                          ── sub_Gsh_prehol (Public 0.146211) ──
                                         │
                       + PLAVE 저녁 공연 보정 (9/12·13 문학)
                       + 작년 같은 시기 실제값 혼합 (평일 10% / 주말 5%)
                                         │
                          ── 두 번째 선택 (Public 0.145681) ──
                                         │
                       + 인접역 피처 LightGBM (seed 3개 평균) 10%
                                         │
                          ── 최종 제출 (Public 0.145536) ──
```

행사 보정이 들어간 행(약 2,400행)은 마지막 두 혼합에서 제외한다.

## 2. 구성 요소 설명

### 2.1 공통 모델 파이프라인 (`1_front/src/sincon_v2.py`)

- 여러 기준일(anchor)에서 “기준일까지의 수준 + n일 뒤 변화”를 학습하는 시간대별 회귀. 타깃은 log1p(승차).
- 특징 묶음: base(역·시간·요일·공휴일·연휴 전후), newst(신규역), graph(노선 그래프), event(경기 일정), analog(작년 같은 시기), trend(역별 추세), vac(방학·학기), wx(날씨), nbr(인접역 수준·추세, 마지막 단계 모델만).
- Chronos-2(`amazon/chronos-2`)는 추가 학습 없이 과거 승차와 미래 공변량(달력·방학·경기·날씨)으로 예측.

### 2.2 모델 A — v8dh17 (`1_front/`)

- LightGBM 4개(num_leaves 255/63), XGBoost 4개(depth 10/8), Chronos-2 2개, 신경망 5개 평균. 트리 0.65 : Chronos 0.35, 신경망 0.3 혼합.
- 2024 하반기 합성 자료(`1_front/src/syn2024.py`): data.go.kr 2024-07~2025-06 역별 일 승차 실측 + 2024 이전 시간대 비율로 2024-07~12 시간대 승차를 만들어 학습 기간을 늘림. 2026-07 이후 자료는 쓰지 않음.
- 신규역(3107~3109)은 합성 자료 없이 학습한 판 사용.

### 2.3 모델 B — v10 (`0_v10/`)

- 0.8 × [LightGBM(lt1·lt2) 0.5 + Chronos-2(chr_e3·wx_chr_st) 0.5] + 0.2 × 가벼운 LightGBM, v8d 후처리.

### 2.4 블렌드와 보정

| 순서 | 보정 | 하는 일 | 코드 |
| --- | --- | --- | --- |
| 1 | v8d 후처리 | 날씨(강수), 야구·축구 경기, 명절 유사일, 공연·축제, 신규역 | `1_front/src/final_post.py` |
| 2 | 행사·추세 보정 12개 | 펜타포트, 7~8월 평일 수준, 아시아드 이상치, 신규역 추세, PSY, 컨벤시아, 아트센터·트라이보울 공연, 부천FC(관중 수), 인천대공원 가을 연휴, 송도해변축제 제거·능허대, 추석 성묘, 맥주축제 | `0b_our_corrections/` |
| 3 | 수준 조정 | 전체 −1%, 7월 +3%, 9월 −2%, 22~23시 −3%, 10~16시 +1% | `1_front/src/lb_chain.py` |
| 4 | 폭염 | 최고기온 30℃ 초과 1℃당 −0.57% | `1_front/src/heat_post.py` |
| 5 | 추석 귀성일 | 9/23 시간대 패턴(과거 연휴 직전 평일 3회 평균 × 0.7) | `1_front/src/prehol_post.py` |
| 6 | PLAVE 공연 | 22시 정점형 저녁 공연(과거 4회) 역×시간 편차 | `2_final_steps/code/post_concert.py` |
| 7 | 작년 같은 시기 혼합 | 2025년 같은 요일 5주 중앙값 × 전체 평일 성장률, 평일 10%·주말 5% (신규역·공휴일·추석 전후·경기장 주변역 제외) | `2_final_steps/code/post_lyana2.py` |
| 8 | 인접역 LightGBM | lt1 설정 + nbr 특징, seed 0·2·3 평균 10% | `2_final_steps/code/train_lgb.py`, `mix_models.py` |

## 3. 검증 방법

시간 순서를 지킨 검증 구간 (`1_front/src/sincon_v2.py` FOLDS). 특징은 각
구간의 학습 기간 데이터로만 만든다.

| 구간 | 학습 | 검증 | 의미 |
| --- | --- | --- | --- |
| fold A | ~2025-06-30 | 2025-07-01 ~ 10-31 | 예측 기간과 같은 계절 |
| fold B | ~2026-02-28 | 2026-03-01 ~ 06-30 | 가장 최근 추세 |
| fold C | ~2025-10-31 | 2025-11-01 ~ 2026-02-28 | 겨울·명절 (일부 실험) |

| 모델 (기존 역, 후처리 포함) | fold A | fold B | Public |
| --- | --- | --- | --- |
| v8d | 0.1987 | 0.1739 | 0.15475 |
| v10 | 0.1976 | 0.1699 | 0.153311 |
| v8dh13 (sub_BASE) | 0.1904 | 0.1628 | 0.152174 |
| v8dh17 | 0.1837 (합성 자료가 A 기간을 일부 봄) | 0.1601 | — |

- 행사 보정은 행사별로 과거 같은 종류 행사를 하나씩 빼고 맞히는 방식(leave-one-event-out)으로 크기를 정했다.
- 작년 같은 시기 혼합은 fold B 에서 개선(−0.002)을 확인했다.
- 검증 예측: `records_validation/`

## 4. 사용 데이터

- 대회 제공: `train.csv`(2025-01-01 ~ 2026-06-30 승하차), `test.csv`, `metaData.csv` (변형 없음).
- 외부 자료 (모두 공개 자료):
    - 시간별 날씨 (Open-Meteo, 2024~2026-10)
    - SSG·부천FC 홈경기 일정, SSG 관중 수·종료 시각 (KBO·K리그 공개 기록)
    - 송도컨벤시아·공연·축제 일정, 행사 취소 보도 (주최 측 공지·보도)
    - 역 좌표 (공개 지도)
    - 인천교통공사 월별 시간대 승하차 2015~2024, 역별 일 승차 2024-07~2025-06 (공공데이터포털)
    - 서울 버스 일별 승차 지수 (서울 열린데이터광장 OA-12912, 2025-01~2026-10, 서울 시내버스 자료)
- 사전학습 모델: Chronos-2 (가중치 그대로)
- 상세 목록·출처: `1_front/README.md`, `2_final_steps/README.md` 3절

## 5. 실행 방법

### 5.1 환경

| 항목 | 내용 |
| --- | --- |
| 후처리·전체 재현 (`run_all.sh`) | Windows 11, Git Bash, Python 3.13.2, CPU 8코어. `pip install -r 2_final_steps/requirements.txt` (pandas 3.1.1, numpy 2.5.4, lightgbm 4.7.0, pyarrow 26.0.1) |
| 모델 학습 (1_front, 0_v10) | Kaggle Notebook (GPU T4, Python 3.13 기본 이미지), chronos-forecasting 2.3.2 (커널에서 설치) |
| 인접역 LightGBM 재학습 | 위 로컬 환경, CPU (모델당 약 10분) |
| seed | 1_front: `1_front/kernels/*/config_head.txt` / 2_final_steps: LightGBM seed 0, 2, 3 |

### 5.2 명령

```bash
pip install -r 2_final_steps/requirements.txt
bash run_all.sh                       # 저장된 모델 예측으로 전체 재현 → 2_final_steps/output/final.csv (약 1분)
bash 2_final_steps/run.sh retrain     # 인접역 LightGBM 3개까지 재학습 (약 30분)
```

| 단계 | 명령 | 출력 | 시간 |
| --- | --- | --- | --- |
| (a) 2024 합성 자료 | (1_front 에서) `python src/syn2024.py make 2026-06-30 full` | `1_front/ext/syn_2024h2_full.csv` | 수 분 |
| (b) 모델 학습 | `1_front/kernels/<모델>/config_head.txt` 의 CONFIG + `1_front/src/sincon_v2.py` 를 Kaggle 커널로 실행, v10 은 `0_v10/캐글노트북/nb.ipynb` | `testpred_<tag>.csv` → `1_front/work/` | 모델당 10분~1시간 |
| (c) 블렌드 입력 | v8dh13·고스티벌·취소경기 처리(`1_front/run_final.sh` 0단계), 0.6×sub_BASE + 0.4×v10, 행사 보정 12개(`0b_our_corrections/run_safe.sh`) | `1_front/teammate/` | 수 분 |
| (d) 앞단 앙상블·보정 | `bash 1_front/run_final.sh` | `1_front/out_repro/sub_Gsh_prehol.csv` | 수 분 |
| (e) 마지막 3단계 | `bash 2_final_steps/run.sh` | `2_final_steps/output/final.csv` | 1분 |

### 5.3 캐시·중간 산출물

| 산출물 | 위치 | 포함 | 재생성 |
| --- | --- | --- | --- |
| 모델 테스트 예측 | `1_front/work/testpred_*.csv`, `0_v10/모델예측/`, `2_final_steps/models/` | 예 | 5.2 (b), `run.sh retrain` |
| 합성 자료 | `1_front/ext/syn_2024h2_*.csv` | 예 | 5.2 (a) |
| 블렌드 입력 | `1_front/teammate/`, `0a_inputs/` | 예 | 5.2 (c) |
| 행사 보정 행 표시 | `2_final_steps/data/evmask.csv` | 예 | `2_final_steps/provenance/make_evmask.py` |
| Chronos-2 가중치 | Hugging Face 캐시 | 아니오 (공개 모델) | 첫 실행 시 자동 다운로드 |
| 학습된 모델 가중치 | 해당 없음 | 해당 없음 | 저장하지 않고 seed 고정 재학습 |

### 5.4 재현 결과

| 확인 | 결과 |
| --- | --- |
| `bash run_all.sh` (저장된 모델 예측 사용) | 157,194행, 제출본과 최대 차이 7e-12 — `logs/3_check.log` |
| 행사 보정 12개 (`run_safe.sh`) | 차이 0.0 — `logs/0_run_safe.log` |
| 블렌드 공식 0.6×sub_BASE + 0.4×v10 | 차이 0.0 |
| sub_BASE 고스티벌·취소경기 처리 재현 | 원본과 문학경기장(3127) 90행 다름 (log RMS 0.004). 원본은 별도 처리. 이후 단계는 원본 파일 사용 |
| 인접역 LightGBM 재학습 | 멀티스레드 합산 순서 때문에 소수 미세 차이 가능 (저장 예측 사용 시 0.0) |

## 6. 제출 자료 위치 (운영진 안내 항목별)

| 안내 항목 | 위치 |
| --- | --- |
| 최종 예측 파일 | `predictions/final.csv` (두 번째: `predictions/second_choice_final_v_l5_sh0.csv`) |
| 전체 실행 코드와 순서 | `run_all.sh`, 5.2절, `0_v10/`, `0b_our_corrections/`, `1_front/`, `2_final_steps/` |
| 데이터 출처와 생성 기록 | 4절, `1_front/README.md`, `2_final_steps/README.md`, `1_front/src/syn2024.py` |
| 실행 환경과 방법 | 5절, `2_final_steps/requirements.txt` |
| 설명서 | 이 문서, 하위 README |
| 모델과 중간 산출물 | 5.3절 |
| 기존 개발 기록 | `records_validation/` (검증 예측), `2_final_steps/records/` (Kaggle 제출 이력, 실험 코드), 각 `logs/` |

제출 전 자체 점검

- [x]  최종 예측 파일과 코드·모델·설정의 관계를 1·2·5절에 적었다.
- [x]  사용한 데이터·모델을 4절과 하위 README 에 적었다.
- [x]  제공 데이터부터 최종 예측까지 명령·의존성·seed 를 5절에 적었다.
- [x]  2026-07-01 이후 실제 지하철 승하차 인원은 열람·확보·사용하지 않았다.

## 7. 팀원 기여

- 장규중 (모델 A·앙상블 담당)
    - v8d 공통 파이프라인(sincon_v2.py, final_post.py)
    - v8dh13·v8dh17 모델 학습 (LightGBM·XGBoost·Chronos-2·신경망)
    - 2024 하반기 합성 자료 생성: [data.go.kr](http://data.go.kr/) 일별 실측과 ICTR 시간대 비율 수집
    - 서울 버스 지수, SSG 관중 자료 수집
    - 모델 A 앙상블 비중, 블렌드·수준 조정·폭염·추석 귀성일 보정
    - 앞단 재현 패키지(1_front/)
- 곽도연 (모델 B 담당)
    - v10 모델 (LightGBM lt1·lt2 + Chronos-2 + 가벼운 LightGBM, MLP 실험)
    - 폴드 A·B 검증 예측과 블렌드 후보 검증
    - 고스티벌 취소·우천취소 경기 처리판
    - v10 Kaggle 노트북(0_v10/)
- 강혜인 (보정·통합 담당)
    - 행사·추세 보정 12개: 행사 일정·관중 자료 수집과 행사별 검증
    - 평일·주말 수준 보정, PLAVE 공연 보정, 작년 같은 시기 혼합
    - 인접역 피처 LightGBM 재학습
    - 공휴일·날씨 지연·분해 모델 등 실험과 검증
    - 최종 블렌드와 제출 파일 확정, 제출 자료 통합
