import argparse, os
import numpy as np
import pandas as pd

VEHICLES = {"car", "truck", "bus", "motorcycle"}
COLS = ["frame", "time_s", "x1", "y1", "x2", "y2", "cx", "cy"]


def fmt(t):
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def circ_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def circ_mean(deg):
    r = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def contact_episodes(t, mask, max_gap=0.35, min_n=3):
    idx = np.where(mask)[0]
    eps = []
    if len(idx) == 0:
        return eps
    s = p = idx[0]
    n = 1
    for i in idx[1:]:
        if t[i] - t[p] > max_gap:
            if n >= min_n:
                eps.append((float(t[s]), float(t[p])))
            s, n = i, 0
        p = i
        n += 1
    if n >= min_n:
        eps.append((float(t[s]), float(t[p])))
    return eps


def win(g, t0, t1):
    return g[(g.time_s >= t0) & (g.time_s <= t1)]


def analyse(g, tc):
    pre = win(g, tc - 1.5, tc - 0.05)
    near = win(g, tc + 0.3, tc + 1.5)
    far = win(g, tc + 1.0, tc + 3.0)
    hp = pre.heading.dropna()
    hq = win(g, tc + 0.1, tc + 1.5).heading.dropna()
    return {
        "pre_ok": len(pre) >= 2,
        "pre_peak": float(pre.speed.max()) if len(pre) >= 2 else np.nan,
        "post_med": float(near.speed.median()) if len(near) >= 2 else np.nan,
        "far_med": float(far.speed.median()) if len(far) >= 2 else np.nan,
        "turn": circ_diff(circ_mean(hp), circ_mean(hq)) if len(hp) >= 2 and len(hq) >= 2 else np.nan,
    }


def others_state(groups, skip, t0, t1, thr):
    n = mv = 0
    for oid, og in groups.items():
        if oid in skip:
            continue
        seg = win(og, t0, t1)
        if len(seg) < 3:
            continue
        n += 1
        if (seg.speed >= thr).mean() >= 0.5:
            mv += 1
    return n, mv


def analyse_pair(a, b, groups, A):
    ga, gb = groups[a], groups[b]
    m = ga[COLS].merge(gb[COLS], on="frame", suffixes=("_a", "_b"))
    if len(m) < 3:
        return None
    ix = (np.minimum(m.x2_a, m.x2_b) - np.maximum(m.x1_a, m.x1_b)).clip(lower=0)
    iy = (np.minimum(m.y2_a, m.y2_b) - np.maximum(m.y1_a, m.y1_b)).clip(lower=0)
    area_a = (m.x2_a - m.x1_a) * (m.y2_a - m.y1_a)
    area_b = (m.x2_b - m.x1_b) * (m.y2_b - m.y1_b)
    iomin = (ix * iy) / np.minimum(area_a, area_b)
    sz = (np.sqrt(area_a) + np.sqrt(area_b)) / 2
    gd = np.hypot(m.cx_a - m.cx_b, m.cy_a - m.cy_b) / sz
    contact = ((iomin >= A.iomin) & (gd <= A.gd)).values
    t = m.time_s_a.values

    clip_start = max(ga.time_s.min(), gb.time_s.min())
    for t0, t1 in contact_episodes(t, contact):
        ra, rb = analyse(ga, t0), analyse(gb, t0)
        rs = [ra, rb]
        pre_peaks = [r["pre_peak"] for r in rs if r["pre_ok"]]
        if pre_peaks and max(pre_peaks) < A.min_speed:
            continue  # both already stationary before contact: queue or parked

        drop = [r["pre_ok"] and r["pre_peak"] >= A.min_speed and r["post_med"] <= 0.5 * r["pre_peak"]
                for r in rs]
        turn = [r["pre_ok"] and r["pre_peak"] >= A.min_speed and r["turn"] >= 45 for r in rs]
        post_known = not (np.isnan(ra["far_med"]) or np.isnan(rb["far_med"]))
        stopped = post_known and ra["far_med"] < A.still_thr and rb["far_med"] < A.still_thr
        aftermath = not any(r["pre_ok"] for r in rs)

        score = 2 * any(drop) + 2 * bool(stopped) + 1 * any(turn)
        if score < 2:
            continue

        mask = (t >= t0) & (t <= t1)
        why = [f"boxes overlapped from {fmt(t0)} (overlap {iomin.values[mask].max():.0%}, "
               f"ground points {gd.values[mask].min():.1f} vehicle-sizes apart)"]
        for vid, r, d in zip((a, b), rs, drop):
            if d:
                why.append(f"ID {vid} speed fell from {r['pre_peak']:.2f} to {r['post_med']:.2f}")
        for vid, r, d in zip((a, b), rs, turn):
            if d:
                why.append(f"ID {vid} changed heading by {r['turn']:.0f} deg")
        if stopped:
            why.append("both vehicles stationary 1-3 s after contact")
        elif not post_known:
            why.append("after-impact motion not visible (clip ends or track lost)")
        if aftermath:
            why.append("impact not visible: clip starts during or after contact")
        n_oth, n_mov = others_state(groups, {a, b}, t0 + 0.5, t0 + 3.0, A.still_thr)
        if n_oth:
            why.append(f"{n_mov} of {n_oth} other vehicles kept moving nearby")

        if score >= 4:
            conf, sev, st, typ = "high", "high", "UNUSUAL", "POSSIBLE_COLLISION"
        elif score == 3:
            conf, sev, st, typ = "medium", "medium", "UNUSUAL", "POSSIBLE_COLLISION"
        else:
            conf, sev, st = "low", "low", "REVIEW"
            typ = "COLLISION_AFTERMATH" if aftermath else "POSSIBLE_COLLISION"

        end = min(t0 + 3.0, float(max(ga.time_s.max(), gb.time_s.max())))
        return dict(start_s=round(t0, 2), end_s=round(end, 2), id=int(a), other_id=int(b),
                    cls=ga.cls.iloc[0], other_cls=gb.cls.iloc[0], type=typ, severity=sev,
                    confidence=conf, status=st, reason="; ".join(why))
    return None


def main(A):
    fr = pd.read_csv(A.frames).sort_values(["id", "time_s"])
    sm = pd.read_csv(A.summary)
    fr["cls"] = fr["id"].map(sm.set_index("id")["class"])
    fr = fr.dropna(subset=["cls"])
    groups = {i: g for i, g in fr[fr.cls.isin(VEHICLES)].groupby("id")}
    ids = sorted(groups)

    events = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ga, gb = groups[a], groups[b]
            if min(ga.time_s.max(), gb.time_s.max()) - max(ga.time_s.min(), gb.time_s.min()) < 0.3:
                continue
            e = analyse_pair(a, b, groups, A)
            if e:
                events.append(e)

    cols = ["event_id", "start_s", "end_s", "id", "other_id", "cls", "type", "severity",
            "confidence", "status", "reason"]
    path = os.path.join(A.out, "events.csv")
    os.makedirs(A.out, exist_ok=True)
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)
    old = old[~old.type.isin(["POSSIBLE_COLLISION", "COLLISION_AFTERMATH"])]
    allv = pd.concat([old, pd.DataFrame(events)], ignore_index=True)
    for c in cols:
        if c not in allv:
            allv[c] = np.nan
    allv = allv.sort_values(["start_s", "id"]).reset_index(drop=True)
    allv["event_id"] = np.arange(1, len(allv) + 1)
    allv[cols].to_csv(path, index=False)

    rows = []
    for _, r in sm.iterrows():
        i = int(r["id"])
        es = allv[(allv.id == i) | (allv.other_id == i)]
        s = "UNUSUAL" if (es.status == "UNUSUAL").any() else ("REVIEW" if (es.status == "REVIEW").any() else "NORMAL")
        rows.append({"id": i, "class": r["class"], "status": s,
                     "events": ", ".join(sorted(set(es.type)))})
    pd.DataFrame(rows).to_csv(os.path.join(A.out, "track_status.csv"), index=False)

    if not events:
        print("No collisions or contact events found.")
    for e in events:
        print(f"[{fmt(e['start_s'])} - {fmt(e['end_s'])}] {e['cls'].capitalize()} ID {e['id']} + "
              f"{e['other_cls'].capitalize()} ID {e['other_id']} | {e['type']} | "
              f"confidence {e['confidence']} | {e['status']}\n    why: {e['reason']}")
    print(f"\n{len(events)} collision event(s). events.csv and track_status.csv updated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="outputs/features_frames.csv")
    p.add_argument("--summary", default="outputs/features_summary.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--still_thr", type=float, default=0.15)
    p.add_argument("--min_speed", type=float, default=0.3)
    p.add_argument("--iomin", type=float, default=0.1, help="min overlap / smaller box area")
    p.add_argument("--gd", type=float, default=1.0, help="max ground-point distance in vehicle sizes")
    main(p.parse_args())