#!/bin/bash
# 최종 제출 재현: 팀원 블렌드(input/) 위에 행사·추세 보정 체인 적용 -> output/blend60_ourfix_safe2.csv
set -e
cd "$(dirname "$0")/data"
I=${1:-../input/submission_blend_base60_v10_40.csv}; OUTF=${2:-../output/blend60_ourfix_safe8.csv}; T=../output/tmp; mkdir -p $T
run(){ echo ">> $1"; python ../code/"$@"; }
mkdir -p $T/v9
run post_v9.py      train.csv test.csv . ../code/sincon_v2.py $I $T/v9   # A: 9/28 추석보정 준PO 오염 제거, C: 9~10월 평일 수준
run post_concert.py train.csv test.csv . ../code/sincon_v2.py $T/v9/v9_A_C100.csv $T/s00.csv 1.0  # D: PLAVE 9/12~13 문학 저녁 공연
run post_penta.py        $T/s00.csv $T/s01.csv 1.0   # 펜타포트 7/31~8/2 (3137~3139)
run post_weekC.py        $T/s01.csv $T/s02.csv 7 1.0  # 7~8월 평일 주간 YoY 수준
run post_asiad_clean.py  $T/s02.csv $T/s02.csv       # 아시아드 3209/3210 야간 이상치 정리
run post_nstrend.py      $T/s02.csv $T/s03.csv 0.8   # 신규역 3107~3109 감쇠 추세
run post_psy.py          $T/s03.csv $T/s04.csv 1.0   # PSY 7/11 아시아드
run post_conv.py         $T/s04.csv $T/s05.csv 0.4   # 송도컨벤시아 전시 (3136)
run post_aci.py          $T/s05.csv $T/s06.csv 0.6   # 아트센터인천·트라이보울 공연
run post_bucheon.py      $T/s06.csv $T/s07.csv 1.0   # 부천FC 홈경기 (관중수 반영)
run post_park.py         $T/s07.csv $T/s08.csv 0.75  # 가을 연휴 인천대공원·운연
run post_beach_nhd.py    $T/s08.csv $T/s09.csv       # 송도해변축제 보정 제거 + 능허대(선학)
run post_cemetery.py     $T/s09.csv $T/s10.csv 0.5   # 추석 당일 부평삼거리(가족공원)
run post_beer.py         $T/s10.csv $T/s11.csv 0.45  # 송도맥주축제 야간
run post_weekend.py      $T/s11.csv $T/s12.csv 45 0.5  # 주말 YoY 수준 (fold B 검증)
run post_lyana.py        $T/s12.csv $OUTF 0.15 brkadj  # 작년 같은 시기 실제 x 역별 성장률 혼합
echo DONE
