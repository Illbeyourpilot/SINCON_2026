#!/bin/bash
# 앞단 재현: 모델 테스트 예측(work/testpred_*.csv) -> 앙상블·보정 -> 팀원 블렌드 -> 제출본
# 사용: bash run_final.sh   (패키지 폴더에서, Python 3.10+, pandas/numpy/lightgbm)
set -e
cd "$(dirname "$0")"; R=$(pwd); W="$R/work"
export SINCON_DIRS="$R/data:$R/ext:$R/ext/syn2024" SINCON_OUT="$W" PYTHONIOENCODING=utf-8
COMMON="FP_LGB_NEW=lt1h_fullb,lt2h_fullb FP_XGB=xh10y_full,xh8y_full,xh10y_s2,xh8y_s3 FP_XGB_NEW=xh10_fullb,xh8_fullb,xh10_s2b,xh8_s3b
 FP_CHR=chr_e3y_full,wx_chry_full FP_CHR_NEW=chr_e3_full,wx_chr_full FP_WT=0.65 FP_GHOST_OFF=0 FP_CANCEL_FIX=0
 FP_CHRHIST=0 FP_SSGBOX=1 FP_BUS=0.4 FP_WL=0.75 FP_NEWTREND=0"
mkdir -p out_repro
# 0) v8dh13 (= sub_BASE.csv) 과 sub_BASE_noghost_nocanc.csv (팀원 블렌드 입력의 우리 쪽 0.6)
V13="FP_LGB=lt1h_full,lt2h_full FP_XGB=xh10_full,xh8_full,xh10_s2,xh8_s3 FP_CHRHIST=0.25 FP_NEWTREND=0.5 FP_WT=0.55 FP_WL=0.6 FP_NN=nn13489_full:0.3 FP_SSGBOX=0.5 FP_BUS=0.75 FP_HOLMATCH=0"
(cd "$W" && env $V13 FP_GHOST_OFF=0 FP_CANCEL_FIX=0 FP_OUT=sub_v8dh13.csv python "$R/src/final_post.py" "$W" > "$R/out_repro/log_v8dh13.txt")
(cd "$W" && env $V13 FP_GHOST_OFF=1 FP_CANCEL_FIX=1 FP_OUT=sub_BASE_noghost_nocanc.csv python "$R/src/final_post.py" "$W" > "$R/out_repro/log_BASE_noghost_nocanc.txt")
cp "$W/sub_v8dh13.csv" "$W/sub_BASE_noghost_nocanc.csv" out_repro/
# 1) 우리 앙상블 + 보정 12단계 (final_post.py)
(cd "$W" && env $COMMON FP_LGB=lt1y_full,lt2y_full,lt1y_s2,lt2y_s3 FP_OUT=cmb_Gs.csv python "$R/src/final_post.py" "$W" > "$R/out_repro/log_final_post_Gs.txt")
(cd "$W" && env $COMMON FP_LGB=lt1yng_full,lt1yng_full,lt1y_s2,lt2y_full,lt2y_s3 FP_OUT=cmb_Gng.csv python "$R/src/final_post.py" "$W" > "$R/out_repro/log_final_post_Gng.txt")
# 2) 팀원 블렌드(우리 0.6 / v10 0.4) + 팀원 보정(weekC 1.0, 나머지 0.65) + 수준 조정 -> 폭염 -> 추석 귀성일
for v in Gs Gng; do
  python src/lb_chain.py "$W/cmb_$v.csv" out_repro/t.csv
  python src/heat_post.py out_repro/t.csv out_repro/t2.csv
  python src/prehol_post.py out_repro/t2.csv out_repro/sub_${v}_prehol.csv
done
mv out_repro/sub_Gs_prehol.csv out_repro/sub_Gsh_prehol.csv
# 3) 안전판: 공개 점수로 정한 수준 조정(전체/7월/9월/심야/낮) 제외
LC_G=0 LC_JUL=0 LC_SEP=0 LC_LN=0 LC_MID=0 python src/lb_chain.py "$W/cmb_Gng.csv" out_repro/t.csv
python src/heat_post.py out_repro/t.csv out_repro/t2.csv
python src/prehol_post.py out_repro/t2.csv out_repro/sub_safe_Gng.csv
rm -f out_repro/t.csv out_repro/t2.csv
echo "완료: out_repro/sub_v8dh13.csv, sub_BASE_noghost_nocanc.csv, sub_Gsh_prehol.csv, sub_Gng_prehol.csv, sub_safe_Gng.csv"
