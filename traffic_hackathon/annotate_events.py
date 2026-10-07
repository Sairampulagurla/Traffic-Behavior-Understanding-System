import cv2
import pandas as pd
import argparse
import os


def main():

    parser = argparse.ArgumentParser(
        description="Create final event-annotated traffic video"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input traffic video"
    )

    parser.add_argument(
        "--tracks",
        default="outputs/tracks.csv"
    )

    parser.add_argument(
        "--events",
        default="outputs/events.csv"
    )

    parser.add_argument(
        "--out",
        default="outputs/final_annotated.mp4"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # READ DATA
    # --------------------------------------------------

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return

    if not os.path.exists(args.tracks):
        print(f"[ERROR] Tracks not found: {args.tracks}")
        return

    if not os.path.exists(args.events):
        print(f"[ERROR] Events not found: {args.events}")
        return

    tracks = pd.read_csv(args.tracks)
    events = pd.read_csv(args.events)

    # --------------------------------------------------
    # VIDEO
    # --------------------------------------------------

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("[ERROR] Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # --------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )
    

    writer = cv2.VideoWriter(
        args.out,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_time = frame_number / fps

        # --------------------------------------------------
        # TRACKS IN CURRENT FRAME
        # --------------------------------------------------

        current_tracks = tracks[
            abs(tracks["time_s"] - current_time)
            < (1.0 / fps)
        ]

        # --------------------------------------------------
        # DRAW VEHICLES
        # --------------------------------------------------

        for _, row in current_tracks.iterrows():

            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])

            track_id = int(row["id"])

            cls = str(row["class"])

            label = f"{cls} #{track_id}"

            # Default box
            thickness = 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                thickness
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # --------------------------------------------------
        # FIND ACTIVE EVENTS
        # --------------------------------------------------

        active_events = events[
            (events["start_s"] <= current_time)
            &
            (events["end_s"] >= current_time)
        ]

        # --------------------------------------------------
        # DISPLAY EVENTS
        # --------------------------------------------------

        y = 35

        for _, event in active_events.iterrows():

            event_type = str(
                event.get("type", "EVENT")
            )

            severity = str(
                event.get("severity", "")
            )

            status = str(
                event.get("status", "")
            )

            event_id = event.get("id", "")

            reason = str(
                event.get("reason", "")
            )

            start = float(
                event.get("start_s", 0)
            )

            end = float(
                event.get("end_s", 0)
            )

            # ----------------------------------------------
            # Event title
            # ----------------------------------------------

            title = (
                f"{event_type} | "
                f"ID {event_id} | "
                f"{severity.upper()}"
            )

            cv2.putText(
                frame,
                title,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y += 28

            time_text = (
                f"{start:.1f}s - {end:.1f}s | "
                f"{status.upper()}"
            )

            cv2.putText(
                frame,
                time_text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            y += 25

            # Keep reason short on video
            if len(reason) > 90:
                reason = reason[:87] + "..."

            cv2.putText(
                frame,
                reason,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            y += 35

        # --------------------------------------------------
        # VIDEO TIME
        # --------------------------------------------------

        minutes = int(current_time // 60)
        seconds = current_time % 60

        time_label = (
            f"{minutes:02d}:{seconds:04.1f}"
        )

        cv2.putText(
            frame,
            time_label,
            (width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        frame_number += 1

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    cap.release()
    writer.release()

    print("=" * 60)
    print("ANNOTATED VIDEO CREATED")
    print("=" * 60)

    print(
        f"[PASS] Output: {args.out}"
    )


if __name__ == "__main__":
    main()