
# ================= 신경망 직접 다중시점 모델 (트리·Chronos 와 다른 계열) =================
# sincon_v2.py 본문 뒤에 붙여 실행 (make_nn_kernel.py). 역x시간 일별 log 승차 시계열 -> 인코더(최근 28일 창,
# 요일유형별 평균, 역·시간 임베딩) + 시점별 디코더(목표일 달력, 과거 계절 변화, 시점) 로 1~123일 뒤를 한 번에 예측.
# 학습 목표는 기준일까지 관측된 값만 사용 (기준일 이후 마스크).
import torch
import torch.nn as nn

W, H_ = CONFIG.get("nn_W", 28), 123
BUS_ = CONFIG.get("nn_bus", False)  # 서울 버스 일별 지수 입력
V3 = CONFIG.get("nn_v3", False)  # 날씨·경기·작년 값 입력 추가
V4 = CONFIG.get("nn_v4", False)
Q_ = CONFIG.get("nn_q", False)  # 분기 전(91일) 값 입력 (Web Traffic 1위)
EMA = CONFIG.get("nn_ema", False)  # 가중치 지수평균 + 마지막 3 에폭 예측 평균 (Web Traffic 1위 체크포인트 평균)
ARCH = CONFIG.get("nn_arch", "mlp")  # mlp | cnn (WaveNet식 팽창 합성곱 인코더, Favorita 2위)  # 공휴일 거리 + 행사장(문학·아시아드·송도) 공연일 입력
# 행사장 공표 일정 (테스트 기간, final_post.EVENTS 와 동일). 학습 기간은 급증 감지로 대신함
NN_EVENTS = {"munhak": ["2026-08-29", "2026-08-30", "2026-09-12", "2026-09-13", "2026-10-24", "2026-10-25"],
             "asiad": ["2026-07-11"],
             "songdo": ["2026-07-31", "2026-08-01", "2026-08-02", "2026-08-08", "2026-08-15", "2026-08-22", "2026-08-23",
                        "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30",
                        "2026-09-12"]}
NN_VENUE = {"munhak": ([3127], "dist_munhak"), "asiad": ([3209], "dist_asiad"), "songdo": ([3139], "dist_songdo_park")}


def nn_calendar(dates):
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    df["dow"] = df.date.dt.dayofweek
    df["daytype"] = np.where(df.date.isin(HSET), 2, np.where(df.dow >= 5, 1, 0))
    df = calendar_feats(df)
    dt4 = np.where(df.date.isin(HSET), 3, np.where(df.dow == 6, 2, np.where(df.dow == 5, 1, 0)))
    f = [np.eye(4)[dt4], np.eye(7)[df.dow.to_numpy()],
         df[["hol_prev", "hol_next", "bridge", "after_long_off", "univ_vac", "school_vac"]].to_numpy(float),
         (np.abs(df.big_hol_rel.to_numpy()) <= 10)[:, None].astype(float),
         np.where(np.abs(df.big_hol_rel.to_numpy()) <= 10, df.big_hol_rel.to_numpy() / 10.0, 0.0)[:, None],
         np.sin(2 * np.pi * df.doy.to_numpy() / 365.25)[:, None], np.cos(2 * np.pi * df.doy.to_numpy() / 365.25)[:, None],
         (np.nan_to_num(df.univ_term_day.to_numpy(), nan=60.0) / 60.0)[:, None]]
    if V4:
        hs_ = np.array(sorted(HSET), dtype="datetime64[D]")
        dd = df.date.to_numpy().astype("datetime64[D]")
        nx = np.searchsorted(hs_, dd)
        to_next = np.where(nx < len(hs_), (hs_[np.minimum(nx, len(hs_) - 1)] - dd).astype(int), 30)
        pv = np.searchsorted(hs_, dd, side="right") - 1
        from_prev = np.where(pv >= 0, (dd - hs_[np.maximum(pv, 0)]).astype(int), 30)
        f += [(np.minimum(to_next, 14) / 14.0)[:, None], (np.minimum(from_prev, 14) / 14.0)[:, None]]
    return np.concatenate(f, axis=1).astype(np.float32), dt4, df.date.dt.month.to_numpy() - 1


class Net(nn.Module):
    def __init__(self, n_st, n_cal, n_x=1, d=192):
        super().__init__()
        self.se, self.he = nn.Embedding(n_st, 16), nn.Embedding(18, 8)
        n_in = W * 2 + W * 4 + 8 + 16 + 8
        self.enc = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d), nn.GELU())
        if ARCH == "cnn":
            ch = 64
            self.conv = nn.ModuleList([nn.Conv1d(6 if i == 0 else ch, ch, 3, dilation=dl, padding=dl)
                                       for i, dl in enumerate((1, 2, 4, 8, 16))])
            self.enc = nn.Sequential(nn.Linear(ch * 2 + 8 + 16 + 8, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(d + n_cal + n_x + 1 + 16 + 8, d), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, xw, s, h, cal, hs):
        if ARCH == "cnn":
            B0 = xw.shape[0]
            wz, mk, ohz = xw[:, :W], xw[:, W:2 * W], xw[:, 2 * W:6 * W].reshape(B0, W, 4).transpose(1, 2)
            x = torch.cat([wz[:, None], mk[:, None], ohz], 1)
            for i, cv in enumerate(self.conv):
                y_ = torch.nn.functional.gelu(cv(x))
                x = y_ if i == 0 else x + y_
            e = torch.cat([x.mean(2), x[:, :, -1], xw[:, 6 * W:], self.se(s), self.he(h)], 1)
        else:
            e = torch.cat([xw, self.se(s), self.he(h)], 1)
        z = self.enc(e)
        B, K = cal.shape[0], cal.shape[1]
        k = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)[None, :, None].expand(B, K, 1) / H_
        zz = torch.cat([z[:, None, :].expand(B, K, z.shape[1]), cal, hs, k,
                        self.se(s)[:, None, :].expand(B, K, 16), self.he(h)[:, None, :].expand(B, K, 8)], 2)
        return self.dec(zz)[..., 0]


def nn_run(tr, meta, cutoff, pred_days, seed, epochs, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sids = sorted(meta.station_id)
    S = len(sids)
    smap = {s_: i for i, s_ in enumerate(sids)}
    d0 = tr.date.min()
    last = max(pd.Timestamp(cutoff) + pd.Timedelta(days=H_), tr.date.max())
    days = pd.date_range(d0, last + pd.Timedelta(days=1))
    D = len(days)
    cal, dt4, mon = nn_calendar(days)
    Y = np.full((D, S, 18), np.nan, np.float32)
    t = tr[tr.date <= cutoff]
    Y[(t.date - d0).dt.days.to_numpy(), t.station_id.map(smap).to_numpy(), t.hour.to_numpy() - 6] = t.lb.to_numpy()
    Hc = np.nan_to_num(hist_clim(meta)).astype(np.float32)  # [S,18,12]
    ci = (pd.Timestamp(cutoff) - d0).days
    Yt = torch.tensor(Y, device=dev)
    calt, dt4t = torch.tensor(cal, device=dev), torch.tensor(dt4, device=dev)
    mont, Hct = torch.tensor(mon, device=dev), torch.tensor(Hc, device=dev)
    if V3:
        wx = pd.read_csv(_find("weather_incheon_hourly24.csv" if CONFIG.get("syn2024") else "weather_incheon_hourly.csv"), parse_dates=["date"]).sort_values(["date", "hour"])
        wx["p3"] = wx.precipitation.rolling(3, min_periods=1).sum()
        wx = wx[wx.hour.between(6, 23) & wx.date.between(days[0], days[-1])]
        WX = np.zeros((D, 18, 4), np.float32)
        di, hi = (wx.date - d0).dt.days.to_numpy(), wx.hour.to_numpy() - 6
        WX[di, hi] = np.stack([np.log1p(wx.precipitation), np.log1p(wx.p3), wx.temperature_2m / 30.0,
                               wx.wind_speed_10m / 10.0], 1)
        EV = np.zeros((D, 18, 4), np.float32)
        hrs = np.arange(6, 24)
        for j, kind in enumerate(("ssg", "bucheon")):
            g = pd.read_csv(_find(ev_file(kind)), parse_dates=["date"])
            g = g[g.date.between(days[0], days[-1])]
            for dt_, st_ in zip(g.date, g.start_hour):
                EV[(dt_ - d0).days, :, 2 * j] = 1.0
                EV[(dt_ - d0).days, :, 2 * j + 1] = np.clip((hrs - st_) / 6.0, -1, 1.5)
        geo = pd.read_csv(_find("station_geo_graph.csv"), encoding="utf-8-sig").set_index("station_id").reindex(sids)
        MW = np.exp(-geo.dist_munhak.to_numpy() / 1000.0).astype(np.float32)
        BW = np.exp(-geo.dist_bucheon_stadium.to_numpy() / 1000.0).astype(np.float32)
        WXt, EVt = torch.tensor(WX, device=dev), torch.tensor(EV, device=dev)
        if V4:
            # 행사일: 학습 기간은 대표역 하루 승차가 앞뒤 3주 중앙값의 1.8배(아시아드 1.4배) 이상(문학은 SSG 홈경기일 제외), 이후는 공표 일정
            VE = np.zeros((D, 3), np.float32)
            VW = np.zeros((S, 3), np.float32)
            ssg = set(pd.read_csv(_find(ev_file("ssg")), parse_dates=["date"]).date)
            for j, (v, (reps, dcol)) in enumerate(NN_VENUE.items()):
                x = tr[tr.station_id.isin(reps)].groupby("date").boardings.sum()
                ratio = x / x.rolling(21, center=True, min_periods=7).median()
                hit = set(ratio[(ratio >= (1.4 if v == "asiad" else 1.8))].index) - (ssg if v == "munhak" else set())
                hit |= set(pd.to_datetime(NN_EVENTS[v]))
                for dt_ in hit:
                    k_ = (pd.Timestamp(dt_) - d0).days
                    if 0 <= k_ < D:
                        VE[k_, j] = 1.0
                VW[:, j] = np.nan_to_num(np.exp(-geo[dcol].to_numpy() / 1000.0))
                p(f"  행사장 {v}: 행사일 {len(hit)}")
            VEt, VWt = torch.tensor(VE, device=dev), torch.tensor(VW, device=dev)
        MWt, BWt = torch.tensor(np.nan_to_num(MW), device=dev), torch.tensor(np.nan_to_num(BW), device=dev)
    if BUS_:
        # 서울 버스 일별 이동량 지수 (src/bus_day.py, 서울 열린데이터광장 OA-12912)
        bi = pd.read_csv(_find("seoul_bus_index24.csv" if CONFIG.get("syn2024") else "seoul_bus_index.csv"), parse_dates=["date"]).set_index("date").bus_idx
        BUSt = torch.tensor(np.nan_to_num(pd.Series(days).map(bi).clip(-0.5, 0.5).to_numpy(dtype=np.float32)), device=dev)

    def batch(a, s, h):
        idx = a[:, None] + torch.arange(-W + 1, 1, device=dev)[None, :]
        w = Yt[idx, s[:, None], h[:, None]]
        m = ~torch.isnan(w)
        mu = torch.nanmean(w, 1)
        dtw = dt4t[idx]
        oh = torch.nn.functional.one_hot(dtw, 4).float()
        wz = torch.where(m, w - mu[:, None], torch.zeros_like(w))
        mean_dt = (oh * wz[..., None]).sum(1) / oh.sum(1).clamp(min=1)
        cnt_dt = (oh * m[..., None].float()).sum(1) / W
        xw = torch.cat([wz, m.float(), (oh * wz[..., None]).reshape(len(a), -1), mean_dt, cnt_dt], 1)
        fk = a[:, None] + torch.arange(1, H_ + 1, device=dev)[None, :]
        hs = (Hct[s[:, None], h[:, None], mont[fk]] - Hct[s, h, mont[a]][:, None])[..., None]
        if V3:
            ly = torch.stack([Yt[(fk - o).clamp(min=0), s[:, None], h[:, None]] for o in (357, 364, 371)], 0)
            ly = torch.where((fk - 371 >= 0)[None], ly, torch.full_like(ly, float("nan")))
            lym = torch.nanmean(ly, 0)
            okl = ~torch.isnan(lym)
            lyz = torch.where(okl, lym - torch.nan_to_num(mu)[:, None], torch.zeros_like(lym))
            ev = EVt[fk, h[:, None]] * torch.stack([MWt[s], MWt[s], BWt[s], BWt[s]], 1)[:, None, :]
            hs = torch.cat([hs, WXt[fk, h[:, None]], ev, lyz[..., None], okl[..., None].float()], 2)
            if V4:
                hs = torch.cat([hs, VEt[fk] * VWt[s][:, None, :]], 2)
            if Q_:
                q = Yt[(fk - 91).clamp(min=0), s[:, None], h[:, None]]
                q = torch.where(fk - 91 >= 0, q, torch.full_like(q, float("nan")))
                okq = ~torch.isnan(q)
                hs = torch.cat([hs, torch.where(okq, q - torch.nan_to_num(mu)[:, None], torch.zeros_like(q))[..., None],
                                okq[..., None].float()], 2)
        if BUS_:
            hs = torch.cat([hs, BUSt[fk][..., None]], 2)
        return xw, calt[fk], hs, mu, fk, m.sum(1)

    net = Net(S, cal.shape[1], 1 + (10 + (3 if V4 else 0) + (2 if Q_ else 0) if V3 else 0) + (1 if BUS_ else 0), d=CONFIG.get("nn_d", 192)).to(dev)
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.998)) if EMA else None
    g_pred = torch.tensor(np.array(np.meshgrid([ci], np.arange(S), np.arange(18), indexing="ij")).reshape(3, -1).T, device=dev)

    def predict(model):
        model.eval()
        with torch.no_grad():
            xw, c, hs, mu, fk, nobs = batch(g_pred[:, 0], g_pred[:, 1], g_pred[:, 2])
            return (model(xw, g_pred[:, 1], g_pred[:, 2], c, hs) + torch.nan_to_num(mu)[:, None]).cpu().numpy()
    snaps = []
    A = np.arange(W + 6, ci)  # 기준일 후보 (목표 1일 이상 관측)
    grid = np.array(np.meshgrid(A, np.arange(S), np.arange(18), indexing="ij")).reshape(3, -1).T
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 1024
    steps = epochs * (len(grid) // 2 // bs)
    if CONFIG.get("nn_smoke"):
        steps = 5
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    gt = torch.tensor(grid, device=dev)
    it = 0
    for ep in range(epochs):
        perm = torch.randperm(len(gt), device=dev)[: len(gt) // 2]
        tot, n = 0.0, 0
        net.train()
        for b in range(0, len(perm) - bs + 1, bs):
            if it >= steps:
                break
            g = gt[perm[b:b + bs]]
            a, s, h = g[:, 0], g[:, 1], g[:, 2]
            xw, c, hs, mu, fk, nobs = batch(a, s, h)
            y = Yt[fk.clamp(max=ci), s[:, None], h[:, None]]
            mk = (fk <= ci) & ~torch.isnan(y) & (nobs[:, None] >= 14) & ~torch.isnan(mu[:, None])
            if mk.sum() == 0:
                continue
            out = net(xw, s, h, c, hs) + torch.nan_to_num(mu)[:, None]
            loss = ((out - torch.nan_to_num(y)) ** 2 * mk).sum() / mk.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            if ema is not None:
                ema.update_parameters(net)
            it += 1
            tot += loss.item()
            n += 1
        p(f"  nn seed{seed} ep{ep} loss {tot / max(n, 1):.4f}")
        if ema is not None and ep >= epochs - 3:
            snaps.append(predict(ema.module))
    out = np.mean(snaps, 0) if snaps else predict(net)
    g = g_pred
    k = np.arange(1, H_ + 1)
    res = pd.DataFrame({"date": np.tile(days[ci + k].to_numpy(), len(g)),
                        "station_id": np.repeat(np.array(sids)[g[:, 1].cpu().numpy()], H_),
                        "hour": np.repeat(g[:, 2].cpu().numpy() + 6, H_), "pred": out.ravel()})
    return res[res.date.isin(pd.DatetimeIndex(pred_days))]


def nn_main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p("장치", dev)
    tr, te, meta, geo = load()
    seeds, ep = CONFIG.get("nn_seeds", [0]), CONFIG.get("nn_epochs", 8)
    for F in CONFIG.get("nn_folds", ["A", "B"]):
        a, b = FOLDS[F]
        days = pd.date_range(pd.Timestamp(a) + pd.Timedelta(days=1), b)
        r = None
        for sd in seeds:
            x = nn_run(tr, meta, a, days, sd, ep, dev)
            r = x if r is None else r.assign(pred=r.pred + x.pred.to_numpy())
        r["pred"] /= len(seeds)
        m = r.merge(tr[["date", "station_id", "hour", "lb"]], on=["date", "station_id", "hour"])
        p(f"[{F}] nn rmse {np.sqrt(np.mean((m.pred.clip(0) - m.lb) ** 2)):.5f} (행 {len(m)})")
        r.to_csv(OUT / f"cv_{CONFIG['tag']}_{F}.csv", index=False)
    if CONFIG.get("nn_full", True):
        days = pd.date_range("2026-07-01", "2026-10-31")
        r = None
        for sd in seeds:
            x = nn_run(tr, meta, "2026-06-30", days, sd, ep, dev)
            r = x if r is None else r.assign(pred=r.pred + x.pred.to_numpy())
        r["pred"] /= len(seeds)
        o = te.merge(r, on=["date", "station_id", "hour"], how="left")
        p("test 결측", int(o.pred.isna().sum()))
        o[["example_id"]].assign(pred_log=o.pred.clip(0).to_numpy()).to_csv(OUT / f"testpred_{CONFIG['tag']}_full.csv", index=False)


nn_main()
