"""SSG 문학 홈경기 박스 정보(관중 수, 시작/종료 시각) 수집 -> ext/syn2024/ssg_box_2024.csv. 출처: koreabaseball.com GameCenter (2026-11-13 접근)."""
import glob, json, re, time, urllib.request, urllib.parse
import pandas as pd
ids = []
for f in sorted(glob.glob("ext/kbo/ssg_2024_*.json")):
    for m in re.finditer(r"gameId=(\d{8}\w\wSK\d)", open(f, encoding="utf-8").read()):
        ids.append(m.group(1))
ids = sorted(set(ids))
rows = []
for g in ids:
    data = urllib.parse.urlencode(dict(leId=1, srId=0, seasonId=g[:4], gameId=g)).encode()
    req = urllib.request.Request("https://www.koreabaseball.com/ws/Schedule.asmx/GetScoreBoardScroll", data=data, headers={
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx"})
    d = {}
    for _ in range(3):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")); break
        except Exception:
            time.sleep(2)
    rows.append(dict(game_id=g, date=d.get("G_DT"), stadium=d.get("S_NM"), crowd=d.get("CROWD_CN"), start=d.get("START_TM"),
                     end=d.get("END_TM"), use=d.get("USE_TM")))
    time.sleep(0.3)
b = pd.DataFrame(rows)
b["crowd"] = pd.to_numeric(b.crowd.astype(str).str.replace(",", ""), errors="coerce")
b.to_csv("ext/syn2024/ssg_box_2024.csv", index=False)
print(len(b), b.stadium.value_counts().to_dict(), "결측", b.end.isna().sum()); print(b.head(3).to_string()); print(b.tail(3).to_string())
