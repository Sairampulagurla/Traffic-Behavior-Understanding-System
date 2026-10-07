import os, csv, argparse
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

# COCO ids: person, bicycle, car, motorcycle, bus, truck, backpack, handbag, suitcase
DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7, 24, 26, 28]


def color_for(track_id):
    # stable, distinct color per ID
    return ((track_id * 53) % 255, (track_id * 97) % 255, (track_id * 29) % 255)


def run(video, weights="yolo11s.pt", conf=0.35, stride=2,
        classes=DEFAULT_CLASSES, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(weights)
    names = model.names

    out_video = os.path.join(out_dir, "annotated.mp4")
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (W, H))

    rows = []
    trails = defaultdict(lambda: deque(maxlen=40))

    results = model.track(
        source=video, stream=True, persist=True,
        tracker="bytetrack.yaml", conf=conf, classes=classes,
        vid_stride=stride, verbose=False,
    )

    for i, r in enumerate(results):
        frame = r.orig_img.copy()
        frame_idx = i * stride
        t = frame_idx / fps  # time in seconds

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            cls = r.boxes.cls.int().tolist()
            confs = r.boxes.conf.tolist()
            boxes = r.boxes.xyxy.tolist()

            for tid, c, cf, (x1, y1, x2, y2) in zip(ids, cls, confs, boxes):
                cx, cy = (x1 + x2) / 2, y2  # bottom-center = ground point
                trails[tid].append((int(cx), int(cy)))
                rows.append([frame_idx, round(t, 2), tid, names[c],
                             round(cf, 2), int(x1), int(y1), int(x2), int(y2),
                             int(cx), int(cy)])

                col = color_for(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(frame, f"{names[c]} #{tid}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                pts = list(trails[tid])
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(frame, a, b, col, 2)

        cv2.putText(frame, f"t={t:6.1f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()

    out_csv = os.path.join(out_dir, "tracks.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "id", "class", "conf",
                    "x1", "y1", "x2", "y2", "cx", "cy"])
        w.writerows(rows)

    print(f"Saved: {out_video}\nSaved: {out_csv}\nFPS={fps:.1f}, rows={len(rows)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--stride", type=int, default=2)
    a = p.parse_args()
    run(a.video, a.weights, a.conf, a.stride)