import argparse, os
import numpy as np
import pandas as pd


def circ_mean(deg):
    r = np.radians(np.asarray(deg, dtype=float))
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def longest_still(g, thr):
    """Longest continuous stretch with speed below thr. Returns (seconds, start_time)."""
    still = (g["speed"] < thr).fillna(False).values
    t = g["time_s"].values
    best, best_start, start, prev = 0.0, np.nan, None, t[0]
    for s, ti in zip(still, t):
        if s and start is None:
            start = ti
        if not s and start is not None:
            if prev - start > best:
                best, best_start = prev - start, start
            start = None
        prev = ti
    if start is not None and prev - start > best:
        best, best_start = prev - start, start
    return best, best_start


def build(csv_path, out_dir="outputs", min_dur=1.0, still_thr=0.15):
    df = pd.read_csv(csv_path).sort_values(["id", "time_s"])
    df["size"] = np.sqrt((df.x2 - df.x1) * (df.y2 - df.y1))

    tracks, rows = [], []
    for tid, g in df.groupby("id"):
        g = g.copy()
        if g.time_s.max() - g.time_s.min() < min_dur or len(g) < 5:
            continue

        g["sx"] = g.cx.rolling(5, center=True, min_periods=1).mean()
        g["sy"] = g.cy.rolling(5, center=True, min_periods=1).mean()
        dt = g.time_s.diff()
        dx, dy = g.sx.diff(), g.sy.diff()

        # speed in "object sizes per second" so near and far vehicles are comparable
        size = g["size"].rolling(5, min_periods=1).median()
        g["speed"] = (np.hypot(dx, dy) / dt / size).rolling(5, min_periods=1).median()
        g["heading"] = np.degrees(np.arctan2(dy, dx))
        g.loc[g["speed"] < still_thr, "heading"] = np.nan  # heading is meaningless when still
        tracks.append(g)

        still_s, still_t = longest_still(g, still_thr)
        rows.append({
            "id": tid,
            "class": g["class"].mode().iloc[0],
            "t_start": round(g.time_s.min(), 2),
            "t_end": round(g.time_s.max(), 2),
            "duration_s": round(g.time_s.max() - g.time_s.min(), 2),
            "median_speed": round(g.speed.median(), 3),
            "max_speed": round(g.speed.max(), 3),
            "moving_fraction": round((g.speed >= still_thr).mean(), 2),
            "heading_deg": round(circ_mean(g.heading), 1),
            "longest_still_s": round(still_s, 2),
            "still_from_s": round(still_t, 2) if not np.isnan(still_t) else np.nan,
        })

    summary = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(tracks).to_csv(os.path.join(out_dir, "features_frames.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "features_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tracks.csv")
    a = p.parse_args()
    s = build(a.csv)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    print("\nmedian of median speeds:", round(s.median_speed.median(), 3))
    print("dominant heading (deg):", round(circ_mean(s.heading_deg), 1))