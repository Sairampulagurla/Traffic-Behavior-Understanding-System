import argparse, json, os
import numpy as np
import pandas as pd
import cv2


def mad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def unit(deg):
    r = np.radians(np.asarray(deg, float))
    return np.stack([np.cos(r), np.sin(r)], axis=1)


def ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def ang_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def two_means(U, iters=25):
    c0 = U[0]
    c1 = U[np.argmin(U @ c0)]
    C = np.stack([c0, c1])
    for _ in range(iters):
        lab = np.argmax(U @ C.T, axis=1)
        newC = []
        for k in range(2):
            m = U[lab == k]
            if len(m) == 0:
                newC.append(C[k])
                continue
            v = m.mean(axis=0)
            n = np.linalg.norm(v)
            newC.append(v / n if n > 0 else C[k])
        newC = np.array(newC)
        if np.allclose(newC, C):
            break
        C = newC
    return C, np.argmax(U @ C.T, axis=1)


def flow_modes(headings):
    """Return 1 or 2 dominant traffic directions."""
    U = unit(headings)
    mean = U.mean(axis=0)
    modes = [{"angle_deg": round(ang(mean), 1), "share": 1.0,
              "concentration": round(float(np.linalg.norm(mean)), 2)}]
    if len(U) >= 20:
        C, lab = two_means(U)
        shares = [float((lab == k).mean()) for k in range(2)]
        if min(shares) >= 0.2 and ang_diff(ang(C[0]), ang(C[1])) > 90:
            modes = []
            for k in range(2):
                m = U[lab == k].mean(axis=0)
                modes.append({"angle_deg": round(ang(m), 1),
                              "share": round(shares[k], 2),
                              "concentration": round(float(np.linalg.norm(m)), 2)})
    return modes


def nearest_mode(h, modes):
    d = [ang_diff(h, m["angle_deg"]) for m in modes]
    k = int(np.argmin(d))
    return k, float(d[k])


def build(frames_csv, summary_csv, video=None, out_dir="outputs", still_thr=0.15):
    fr = pd.read_csv(frames_csv)
    sm = pd.read_csv(summary_csv)
    fr["cls"] = fr["id"].map(sm.set_index("id")["class"])

    if video:
        cap = cv2.VideoCapture(video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, first = cap.read()
        cap.release()
    else:
        W, H = int(fr.sx.max()) + 50, int(fr.sy.max()) + 50
        ok, first = False, None
    if not ok:
        first = np.zeros((H, W, 3), np.uint8)

    mv = fr[(fr.speed >= still_thr) & fr.heading.notna()].copy()
    n_tracks, n_samples = len(sm), len(mv)
    reliable = n_tracks >= 10 and n_samples >= 200

    stats = {"video_size": [W, H], "n_tracks": int(n_tracks),
             "n_moving_samples": int(n_samples),
             "reliability": "ok" if reliable else "weak (use default rule thresholds)"}

    # 1. Dominant flow directions
    modes = flow_modes(mv.heading.values) if n_samples >= 5 else []
    stats["flow_modes"] = modes
    stats["two_way"] = len(modes) == 2

    # 2. Typical speed per class (robust: median and MAD)
    spd = {}
    for c, g in mv.groupby("cls"):
        if len(g) >= 30:
            med = float(g.speed.median())
            thr = med + 3 * max(mad(g.speed), 0.1 * med)
            spd[c] = {"median": round(med, 3), "mad": round(mad(g.speed), 3),
                      "fast_threshold": round(thr, 3), "samples": int(len(g))}
    stats["speed_by_class"] = spd

    # 3. Typical stops
    stops = sm[sm.longest_still_s >= 1.0].longest_still_s
    if len(stops) >= 3:
        stop_alert = max(3.0, float(stops.median() + 3 * mad(stops)))
        src = "learned"
    else:
        stop_alert, src = 5.0, "default (too few stops to learn)"
    stats["stops"] = {"n_stopped_tracks": int(len(stops)),
                      "median_stop_s": round(float(stops.median()), 2) if len(stops) else None,
                      "stop_alert_s": round(stop_alert, 2), "source": src}

    # 4. Road area and local flow on a grid
    cell = max(32, int(max(W, H) / 30))
    fr["ci"] = (fr.sx // cell).astype(int)
    fr["cj"] = (fr.sy // cell).astype(int)
    counts = fr.groupby(["ci", "cj"]).size()
    min_count = max(3, int(0.02 * counts.max()))
    road = [(int(i), int(j)) for (i, j), n in counts.items() if n >= min_count]
    stats["grid_cell_px"] = cell
    stats["road_cells"] = [list(c) for c in road]
    stats["road_area_fraction"] = round(len(road) / ((W // cell + 1) * (H // cell + 1)), 3)

    mv["ci"] = (mv.sx // cell).astype(int)
    mv["cj"] = (mv.sy // cell).astype(int)
    local = {}
    for (i, j), g in mv.groupby(["ci", "cj"]):
        if len(g) >= 5:
            v = unit(g.heading.values).mean(axis=0)
            local[(int(i), int(j))] = (ang(v), float(np.linalg.norm(v)), int(len(g)))
    stats["local_flow"] = {f"{i},{j}": [round(a, 1), round(c, 2), n]
                           for (i, j), (a, c, n) in local.items()}

    # 5. Preview of per-track direction (the real rule comes in Step 4)
    prev = []
    for _, r in sm.iterrows():
        if modes and not np.isnan(r.heading_deg):
            k, d = nearest_mode(r.heading_deg, modes)
            prev.append({"id": int(r["id"]), "class": r["class"],
                         "heading": r.heading_deg, "flow": k,
                         "deviation_deg": round(d, 1)})
    stats["track_preview"] = prev

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scene_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # overview image
    ov = first.copy()
    for (i, j) in road:
        cv2.rectangle(ov, (i * cell, j * cell), ((i + 1) * cell, (j + 1) * cell), (0, 200, 0), -1)
    img = cv2.addWeighted(ov, 0.35, first, 0.65, 0)
    for (i, j), (a, c, n) in local.items():
        if c >= 0.6:
            cx, cy = int((i + .5) * cell), int((j + .5) * cell)
            r = np.radians(a)
            cv2.arrowedLine(img, (cx, cy),
                            (int(cx + cell * .4 * np.cos(r)), int(cy + cell * .4 * np.sin(r))),
                            (0, 0, 255), 2, tipLength=0.4)
    cv2.imwrite(os.path.join(out_dir, "scene_overview.png"), img)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="outputs/features_frames.csv")
    p.add_argument("--summary", default="outputs/features_summary.csv")
    p.add_argument("--video", default=None)
    a = p.parse_args()
    s = build(a.frames, a.summary, a.video)

    print("Tracks:", s["n_tracks"], "| moving samples:", s["n_moving_samples"],
          "| reliability:", s["reliability"])
    print("Two-way road:", s["two_way"])
    for m in s["flow_modes"]:
        print("  flow direction", m["angle_deg"], "deg | share", m["share"],
              "| concentration", m["concentration"])
    print("Speed baseline:", json.dumps(s["speed_by_class"]))
    print("Stops:", s["stops"])
    print("Road area fraction:", s["road_area_fraction"])
    print("\nPer-track direction preview:")
    for t in s["track_preview"]:
        print("  ", t)
    print("\nSaved: outputs/scene_stats.json and outputs/scene_overview.png")