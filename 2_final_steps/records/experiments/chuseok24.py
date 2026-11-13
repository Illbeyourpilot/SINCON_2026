"""2024 추석(STCIS) 시간대별 '명절일 / 평소 평일' 비율로 추석 예측 보정.
2024: 9/16(월, 전날) 9/17(화, 당일) 9/18(수, 다음날), 평소 평일 = 9/9~9/12 시간대 중앙값
검증(cv): 2025 추석 10/6(당일) 10/7(다음날) — fold A 예측 P 와 'P 주변 평일 기준 x 2024 비율' 혼합
적용(apply): 2026 9/24(전날) 9/25(당일) — 최종 파일 주변 평일 기준 x 2024 비율 혼합
사용: python chuseok24.py cv | python chuseok24.py apply <in.csv> <out.csv> <w>
"""
import sys, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
NAME2ID = {"인천대공원": 3226, "귤현": 3111, "계양": 3110, "운연": 3227, "검단오류": 3201, "부평삼거리": 3122, "부천종합운동장": 3754,
           "지식정보단지": 3135, "국제업무지구": 3138, "센트럴파크": 3137, "왕길": 3202, "주안국가산단": 3217}
R = {}  # station -> DataFrame(index hour, columns eve/day/after) log ratio
for fn in glob.glob("stcis/2024_*.csv"):
    nm = os.path.basename(fn)[5:-4]
    if nm not in NAME2ID or (nm=="인천대공원" and len(sys.argv)>1 and sys.argv[1]=="apply"): continue
    s = pd.read_csv(fn, encoding="cp949")
    s["date"] = pd.to_datetime(s["일자"].str[:10])
    cols = [c for c in s.columns if "(승차)" in c]
    m = s.set_index("date")[cols]; m.columns = [int(c[:2]) for c in cols]; m = m[[h for h in range(6, 24)]]
    lb = np.log1p(m)
    base = lb.loc["2024-09-09":"2024-09-12"].median()
    R[NAME2ID[nm]] = pd.DataFrame({k: lb.loc[d] - base for k, d in [("eve", "2024-09-16"), ("day", "2024-09-17"), ("after", "2024-09-18")]})
print("stations", sorted(R))


def apply_ratio(df, pcol, mapping, nb_dates):
    """df: date, station_id, hour, pcol(log). mapping: {date: key}. 기준 = nb_dates 의 같은 역·시간 pcol 중앙값"""
    base = df[df.date.isin(nb_dates)].groupby(["station_id", "hour"])[pcol].median()
    alt = np.full(len(df), np.nan)
    for d, key in mapping.items():
        for s, rr in R.items():
            m = ((df.date == d) & (df.station_id == s)).to_numpy()
            if not m.any(): continue
            idx = pd.MultiIndex.from_frame(df.loc[m, ["station_id", "hour"]])
            alt[m] = base.reindex(idx).to_numpy() + rr[key].reindex(df.loc[m, "hour"]).to_numpy()
    return alt


if sys.argv[1] == "cv":
    exec(open("stack_cv.py", encoding="utf-8").read().split("if __name__")[0])
    d = load_fold("A"); d["P"], _, _ = v8d_blend(d)
    nb = pd.to_datetime(["2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02"])
    mp = {pd.Timestamp("2025-10-06"): "day", pd.Timestamp("2025-10-07"): "after"}
    alt = apply_ratio(d, "P", mp, nb)
    ok = ~np.isnan(alt); y = d.lb.to_numpy(); P = d.P.to_numpy()
    print("rows", ok.sum())
    for w in (0, 0.25, 0.5, 0.75, 1.0):
        Q = np.where(ok, (1 - w) * P + w * alt, P)
        print(f"w{w}: rmsle on these rows {r(Q[ok], y[ok]):.4f}")
    t = d[ok].assign(P_=P[ok], alt=alt[ok]); t["eP"] = (t.P_ - t.lb) ** 2; t["eA"] = (0.5 * t.P_ + 0.5 * t.alt - t.lb) ** 2
    print(t.groupby(["station_id", "date"])[["eP", "eA"]].mean().round(3).to_string())
else:
    inp, out, w = sys.argv[2], sys.argv[3], float(sys.argv[4])
    te = pd.read_csv("test.csv", parse_dates=["date"]).merge(pd.read_csv(inp), on="example_id"); te["P"] = np.log1p(te.boardings)
    nb = pd.to_datetime(["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"])
    mp = {pd.Timestamp("2026-09-24"): "eve", pd.Timestamp("2026-09-25"): "day"}
    alt = apply_ratio(te, "P", mp, nb); ok = ~np.isnan(alt)
    if "nocem" in sys.argv: ok &= ~((te.station_id == 3122) & (te.date == "2026-09-25")).to_numpy()
    Q = np.where(ok, (1 - w) * te.P + w * alt, te.P)
    pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(np.clip(Q, 0, None))}).to_csv(out, index=False)
    print("rows", ok.sum(), te[ok].assign(o=np.expm1(te.P[ok]), n=np.expm1(Q[ok])).groupby(["station_id", "date"])[["o", "n"]].sum().round(0).to_string())
