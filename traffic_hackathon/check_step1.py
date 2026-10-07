import pandas as pd

df = pd.read_csv("outputs/tracks.csv")
print(df["class"].value_counts())

d = df.groupby("id").time_s.agg(["min", "max"])
d["dur"] = d["max"] - d["min"]
print("unique IDs:", len(d))
print("median track length (s):", round(d.dur.median(), 1))
print("IDs shorter than 1 s:", (d.dur < 1).sum())