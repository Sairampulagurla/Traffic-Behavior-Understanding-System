import numpy as np, pandas as pd

VEH = {"car", "truck", "bus", "motorcycle"}
fr = pd.read_csv("outputs/features_frames.csv").sort_values(["id", "time_s"])
sm = pd.read_csv("outputs/features_summary.csv")
raw = pd.read_csv("outputs/tracks.csv")
fr["cls"] = fr["id"].map(sm.set_index("id")["class"])

# 1. Were any tracks thrown away?
rng = raw.groupby("id").time_s.agg(["min", "max"]).round(2)
dropped = sorted(set(raw.id) - set(sm.id))
print("IDs in tracks.csv  :", sorted(raw.id.unique()))
print("IDs kept (features):", sorted(sm.id.unique()))
print("Dropped as too short:", {i: (rng.loc[i, "min"], rng.loc[i, "max"]) for i in dropped}, "\n")

# 2. Pair-by-pair closeness
groups = {i: g for i, g in fr[fr.cls.isin(VEH)].groupby("id")}
rows = []
ids = sorted(groups)
for k, a in enumerate(ids):
    for b in ids[k + 1:]:
        ga, gb = groups[a], groups[b]
        m = ga.merge(gb, on="frame", suffixes=("_a", "_b"))
        if len(m) < 3:
            continue
        ix = (np.minimum(m.x2_a, m.x2_b) - np.maximum(m.x1_a, m.x1_b)).clip(lower=0)
        iy = (np.minimum(m.y2_a, m.y2_b) - np.maximum(m.y1_a, m.y1_b)).clip(lower=0)
        aa = (m.x2_a - m.x1_a) * (m.y2_a - m.y1_a)
        ab = (m.x2_b - m.x1_b) * (m.y2_b - m.y1_b)
        iomin = (ix * iy) / np.minimum(aa, ab)
        sz = (np.sqrt(aa) + np.sqrt(ab)) / 2
        gd = np.hypot(m.cx_a - m.cx_b, m.cy_a - m.cy_b) / sz
        j = gd.values.argmin()
        t = m.time_s_a.values[j]

        def sp(g, t0, t1, f):
            s = g[(g.time_s >= t0) & (g.time_s <= t1)].speed
            return round(float(f(s)), 2) if len(s) >= 2 else None

        rows.append(dict(a=a, b=b, t_closest=round(t, 2),
                         min_gd=round(float(gd.min()), 2),
                         max_overlap=round(float(iomin.max()), 2),
                         a_before=sp(ga, t - 1.5, t - .05, np.max),
                         a_after=sp(ga, t + .3, t + 1.5, np.median),
                         b_before=sp(gb, t - 1.5, t - .05, np.max),
                         b_after=sp(gb, t + .3, t + 1.5, np.median)))
out = pd.DataFrame(rows).sort_values("min_gd")
print(out.to_string(index=False) if len(out) else "No vehicle pairs overlap in time at all.")
print("\nCollision rule needs: max_overlap >= 0.10 AND min_gd <= 1.0 AND at least one before-speed >= 0.3")