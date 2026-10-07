import os
import sys
import subprocess


def run_step(command, name):
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n[FAILED] {name}")
        sys.exit(result.returncode)

    print(f"\n[PASS] {name}")


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python analyze.py video.mp4")
        sys.exit(1)

    video = sys.argv[1]

    if not os.path.exists(video):
        print(f"[ERROR] Video not found: {video}")
        sys.exit(1)

    print("=" * 60)
    print("TRAFFIC BEHAVIOR ANALYSIS SYSTEM")
    print("=" * 60)

    print(f"Video: {video}")

    # --------------------------------------------------
    # STEP 1: Detection + Tracking
    # --------------------------------------------------
    run_step(
        [
            sys.executable,
            "detect_track.py",
            video
        ],
        "Detection + Tracking"
    )

    # --------------------------------------------------
    # STEP 2: Feature Extraction
    # --------------------------------------------------
    run_step(
        [
            sys.executable,
            "features.py"
        ],
        "Feature Extraction"
    )

    # --------------------------------------------------
    # STEP 3: Learn Scene Statistics
    # --------------------------------------------------
    run_step(
        [
            sys.executable,
            "scene.py",
            "--video",
            video
        ],
        "Scene Analysis"
    )

    # --------------------------------------------------
    # STEP 4: Behavior Rules
    # --------------------------------------------------
    run_step(
        [
            sys.executable,
            "rules.py"
        ],
        "Behavior Analysis"
    )

    # --------------------------------------------------
    # STEP 5: Collision Detection
    # --------------------------------------------------
    run_step(
    [
        sys.executable,
        "collision.py",
        "--min_speed",
        "0.15",
        "--gd",
        "2.0"
    ],
    "Collision Detection"
)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print("\nGenerated outputs:")

    outputs = [
        "outputs/tracks.csv",
        "outputs/features_frames.csv",
        "outputs/features_summary.csv",
        "outputs/scene_stats.json",
        "outputs/scene_overview.png",
        "outputs/events.csv",
        "outputs/track_status.csv"
    ]

    for file in outputs:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[MISSING] {file}")


if __name__ == "__main__":
    main()