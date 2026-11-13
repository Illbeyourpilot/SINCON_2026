#!/bin/bash
# 1_front/teammate/output/blend60_ourfix_safe.csv 생성: 팀원 블렌드 + 행사·추세 보정 12개
set -e
cd "$(dirname "$0")/data"
I=${1:-../input/submission_blend_base60_v10_40.csv}; OUTF=${2:-../output/blend60_ourfix_safe.csv}; T=../output/tmp_safe; mkdir -p $T
run(){ echo ">> $1"; python ../code/"$@"; }
run post_penta.py       $I         $T/t1.csv 1.0
run post_weekC.py       $T/t1.csv  $T/t2.csv 7 1.0
run post_asiad_clean.py $T/t2.csv  $T/t2.csv
run post_nstrend.py     $T/t2.csv  $T/t3.csv 0.8
run post_psy.py         $T/t3.csv  $T/t4.csv 1.0
run post_conv.py        $T/t4.csv  $T/t5.csv 0.4
run post_aci.py         $T/t5.csv  $T/t6.csv 0.6
run post_bucheon.py     $T/t6.csv  $T/t7.csv 1.0
run post_park.py        $T/t7.csv  $T/t8.csv 0.75
run post_beach_nhd.py   $T/t8.csv  $T/t9.csv
run post_cemetery.py    $T/t9.csv  $T/t10.csv 0.5
run post_beer.py        $T/t10.csv $OUTF 0.45
echo DONE
