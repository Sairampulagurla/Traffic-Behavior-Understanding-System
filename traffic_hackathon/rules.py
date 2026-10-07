import argparse, json, os
import numpy as np
import pandas as pd

VEHICLES = {"car", "truck", "bus", "motorcycle"}


def mad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def fmt(t):
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def ang_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def runs(mask, t, min_dur, max_gap=0.5):
    """Time intervals where mask is True. Gaps up to max_gap seconds are bridged."""
    tt = np.asarray(t)[np.asarray(mask, bool)]
    out = []
    if len(tt) == 0:
        return out
    s = p = tt[0]
    for x in tt[1:]:
        if x - p > max_gap:
            if p - s >= min_dur:
                out.append((float(s), float(p)))
            s = x
        p = x
    if p - s >= min_dur:
        out.append((float(s), float(p)))
    return out


def ev(tid, cls, t0, t1, typ, sev, status, reason):
    return dict(id=int(tid), cls=cls, start_s=round(float(t0), 2), end_s=round(float(t1), 2),
                type=typ, severity=sev, status=status, reason=reason)


def others_state(groups, tid, t0, t1, thr):
    """How many other vehicles are visible in [t0, t1], and how many of them keep moving."""
    n_oth = n_mov = 0
    for oid, og in groups.items():
        if oid == tid:
            continue
        seg = og[(og.time_s >= t0) & (og.time_s <= t1)]
        if len(seg) < 3:
            continue
        n_oth += 1
        if (seg.speed >= thr).mean() >= 0.5:
            n_mov += 1
    return n_oth, n_mov


def rule_stopped(groups, sc, thr, alert):
    out = []
    med_stop = sc["stops"].get("median_stop_s")
    for tid, g in groups.items():
        cls = g.cls.iloc[0]
        for t0, t1 in runs((g.speed < thr).values, g.time_s.values, alert):
            d = t1 - t0
            n_oth, n_mov = others_state(groups, tid, t0, t1, thr)
            ctx = f" (typical stop here: {med_stop}s)" if med_stop else ""
            if n_oth == 0:
                out.append(ev(tid, cls, t0, t1, "STOPPED_VEHICLE", "low", "UNUSUAL",
                              f"stationary {d:.1f}s, alert after {alert:.1f}s{ctx}; "
                              f"no other vehicles visible to compare"))
            elif n_mov / n_oth >= 0.5:
                sev = "high" if d >= 3 * alert else "medium"
                out.append(ev(tid, cls, t0, t1, "STOPPED_VEHICLE", sev, "UNUSUAL",
                              f"stationary {d:.1f}s, alert after {alert:.1f}s{ctx}; "
                              f"{n_mov} of {n_oth} other vehicles kept moving"))
            else:
                out.append(ev(tid, cls, t0, t1, "QUEUED", "info", "NORMAL",
                              f"stationary {d:.1f}s but {n_oth - n_mov} of {n_oth} "
                              f"other vehicles were also stopped (queue or signal)"))
    return out


def rule_wrong_way(groups, sc, thr, dev_thr=120, min_dur=1.5):
    out = []
    modes, cell, local = sc["flow_modes"], sc["grid_cell_px"], sc["local_flow"]
    if not modes:
        return out
    for tid, g in groups.items():
        g = g[(g.speed >= thr) & g.heading.notna()]
        if len(g) < 5:
            continue
        dev = []
        for sx, sy, h in zip(g.sx, g.sy, g.heading):
            key = f"{int(sx // cell)},{int(sy // cell)}"
            ref = None
            if key in local and local[key][1] >= 0.6 and local[key][2] >= 5:
                ref = local[key][0]
            elif not sc["two_way"] and modes[0]["concentration"] >= 0.6:
                ref = modes[0]["angle_deg"]
            dev.append(ang_diff(h, ref) if ref is not None else 0.0)
        dev = np.array(dev)
        for t0, t1 in runs(dev > dev_thr, g.time_s.values, min_dur):
            m = ((g.time_s >= t0) & (g.time_s <= t1)).values
            out.append(ev(tid, g.cls.iloc[0], t0, t1, "WRONG_WAY", "high", "UNUSUAL",
                          f"heading {dev[m].mean():.0f} deg away from the traffic flow "
                          f"in that part of the road for {t1 - t0:.1f}s"))
    return out


def rule_sudden_stop(groups, sc, thr, window=1.5):
    out = []
    for tid, g in groups.items():
        t, v = g.time_s.values, g.speed.fillna(0).values
        if len(t) < 10:
            continue
        cls = g.cls.iloc[0]
        hi = max(0.3, sc["speed_by_class"].get(cls, {}).get("median", 0.3))
        i = 0
        while i < len(t):
            if v[i] < thr:
                w = (t >= t[i] - window) & (t <= t[i])
                peak = v[w].max()
                if peak >= hi:
                    tp = t[w][v[w].argmax()]
                    n_oth, n_mov = others_state(groups, tid, t[i], t[i] + 1.0, thr)
                    txt = f"speed fell from {peak:.2f} to {v[i]:.2f} in {t[i] - tp:.1f}s"
                    if n_oth >= 2 and n_mov / n_oth < 0.5:
                        out.append(ev(tid, cls, tp, t[i], "BRAKING_IN_QUEUE", "info", "NORMAL",
                                      txt + f"; {n_oth - n_mov} of {n_oth} other vehicles also stopped"))
                    else:
                        out.append(ev(tid, cls, tp, t[i], "SUDDEN_STOP", "medium", "UNUSUAL",
                                      txt + f" while {n_mov} of {n_oth} other vehicles kept moving"))
                    i = int(np.searchsorted(t, t[i] + 2.0))
                    continue
            i += 1
    return out


def rule_fast(fr, groups, sc, thr, min_dur=1.0):
    out = []
    allv = fr[fr.cls.isin(VEHICLES) & (fr.speed >= thr)].speed
    if len(allv) >= 30:
        ref_med = float(allv.median())
        ref_thr = ref_med + 3 * max(mad(allv), 0.1 * ref_med)
    else:
        ref_med = ref_thr = None
    for tid, g in groups.items():
        cls = g.cls.iloc[0]
        info = sc["speed_by_class"].get(cls)
        med = info["median"] if info else ref_med
        lim = info["fast_threshold"] if info else ref_thr
        if lim is None:
            continue
        for t0, t1 in runs(g.speed.fillna(0).values > lim, g.time_s.values, min_dur):
            seg = g[(g.time_s >= t0) & (g.time_s <= t1)]
            pk = seg.speed.max()
            out.append(ev(tid, cls, t0, t1, "SPEEDING", "medium", "UNUSUAL",
                          f"speed up to {pk:.2f} vs typical {med:.2f} for {cls} "
                          f"({pk / med:.1f}x median, threshold {lim:.2f})"))
    return out


def rule_ped_on_road(fr, sc, thr, min_dur=2.0):
    cell = sc["grid_cell_px"]
    veh = fr[fr.cls.isin(VEHICLES) & (fr.speed >= thr)]
    if len(veh) < 100:
        return []
    cnt = veh.groupby([(veh.sx // cell).astype(int), (veh.sy // cell).astype(int)]).size()
    road = {k for k, n in cnt.items() if n >= max(5, int(0.05 * cnt.max()))}
    out = []
    for tid, g in fr[fr.cls == "person"].groupby("id"):
        on = np.array([(int(x // cell), int(y // cell)) in road for x, y in zip(g.sx, g.sy)])
        for t0, t1 in runs(on, g.time_s.values, min_dur):
            d = t1 - t0
            out.append(ev(tid, "person", t0, t1, "PEDESTRIAN_ON_ROAD",
                          "high" if d >= 3 else "medium", "UNUSUAL",
                          f"inside the area where vehicles travel for {d:.1f}s"))
    return out


def main(a):
    fr = pd.read_csv(a.frames).sort_values(["id", "time_s"])
    sm = pd.read_csv(a.summary)
    sc = json.load(open(a.scene))
    fr["cls"] = fr["id"].map(sm.set_index("id")["class"])
    fr = fr.dropna(subset=["cls"])

    thr = a.still_thr
    alert = a.stop_alert if a.stop_alert else sc["stops"]["stop_alert_s"]
    groups = {tid: g for tid, g in fr[fr.cls.isin(VEHICLES)].groupby("id")}

    if sc["reliability"] != "ok":
        print("WARNING: scene statistics are weak (short clip or few vehicles). "
              "Thresholds fall back to defaults, so treat results as a test only.\n")

    events = (rule_stopped(groups, sc, thr, alert)
              + rule_wrong_way(groups, sc, thr)
              + rule_sudden_stop(groups, sc, thr)
              + rule_fast(fr, groups, sc, thr)
              + rule_ped_on_road(fr, sc, thr))
    events.sort(key=lambda e: (e["start_s"], e["id"]))
    for k, e in enumerate(events, 1):
        e["event_id"] = k

    os.makedirs(a.out, exist_ok=True)
    cols = ["event_id", "start_s", "end_s", "id", "cls", "type", "severity", "status", "reason"]
    pd.DataFrame(events, columns=cols).to_csv(os.path.join(a.out, "events.csv"), index=False)

    st = []
    for _, r in sm.iterrows():
        es = [e for e in events if e["id"] == int(r["id"])]
        bad = [e["type"] for e in es if e["status"] == "UNUSUAL"]
        st.append({"id": int(r["id"]), "class": r["class"],
                   "status": "UNUSUAL" if bad else "NORMAL",
                   "events": ", ".join(sorted(set(e["type"] for e in es)))})
    pd.DataFrame(st).to_csv(os.path.join(a.out, "track_status.csv"), index=False)

    print(f"Stop alert threshold: {alert:.1f}s | still threshold: {thr}\n")
    if not events:
        print("No events. Every tracked object behaved within the learned normal range.")
    for e in events:
        print(f"[{fmt(e['start_s'])} - {fmt(e['end_s'])}] {e['cls'].capitalize()} ID {e['id']} "
              f"| {e['type']} | {e['severity']} | {e['status']}\n    why: {e['reason']}")
    n_un = sum(1 for e in events if e["status"] == "UNUSUAL")
    print(f"\n{len(events)} events ({n_un} unusual). Saved events.csv and track_status.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="outputs/features_frames.csv")
    p.add_argument("--summary", default="outputs/features_summary.csv")
    p.add_argument("--scene", default="outputs/scene_stats.json")
    p.add_argument("--out", default="outputs")
    p.add_argument("--still_thr", type=float, default=0.15)
    p.add_argument("--stop_alert", type=float, default=None,
                   help="override the learned stop-alert seconds (useful for short test clips)")
    main(p.parse_args())