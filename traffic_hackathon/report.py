import os
import json
import argparse
import pandas as pd


def safe_value(value):
    """Convert pandas/numpy values into JSON-safe values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def main():

    parser = argparse.ArgumentParser(
        description="Build final traffic analysis result"
    )

    parser.add_argument(
        "--events",
        default="outputs/events.csv"
    )

    parser.add_argument(
        "--status",
        default="outputs/track_status.csv"
    )

    parser.add_argument(
        "--features",
        default="outputs/features_summary.csv"
    )

    parser.add_argument(
        "--scene",
        default="outputs/scene_stats.json"
    )

    parser.add_argument(
        "--out",
        default="outputs/analysis_result.json"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BUILDING TRAFFIC ANALYSIS RESULT")
    print("=" * 60)

    # --------------------------------------------------
    # Read files
    # --------------------------------------------------

    if os.path.exists(args.events):
        events_df = pd.read_csv(args.events)
    else:
        events_df = pd.DataFrame()

    if os.path.exists(args.status):
        status_df = pd.read_csv(args.status)
    else:
        status_df = pd.DataFrame()

    if os.path.exists(args.features):
        features_df = pd.read_csv(args.features)
    else:
        features_df = pd.DataFrame()

    if os.path.exists(args.scene):
        with open(args.scene, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
    else:
        scene_data = {}

    # --------------------------------------------------
    # BASIC SUMMARY
    # --------------------------------------------------

    vehicle_classes = {
        "car",
        "truck",
        "bus",
        "motorcycle"
    }

    person_classes = {
        "person"
    }

    vehicles = 0
    people = 0

    if not features_df.empty and "class" in features_df.columns:

        classes = (
            features_df["class"]
            .astype(str)
            .str.lower()
        )

        vehicles = int(
            classes.isin(vehicle_classes).sum()
        )

        people = int(
            classes.isin(person_classes).sum()
        )

    # Unique IDs are more meaningful than row count
    unique_entities = 0

    if not features_df.empty and "id" in features_df.columns:
        unique_entities = int(
            features_df["id"].nunique()
        )

    # --------------------------------------------------
    # EVENTS
    # --------------------------------------------------

    event_list = []

    if not events_df.empty:

        for _, row in events_df.iterrows():

            event = {}

            for column in events_df.columns:
                event[column] = safe_value(row[column])

            # Make IDs easier for the website
            if "id" in event and event["id"] is not None:
                event["entity_id"] = int(event["id"])

            event_list.append(event)

    # --------------------------------------------------
    # EVENT COUNTS
    # --------------------------------------------------

    event_counts = {}

    for event in event_list:

        event_type = event.get(
            "type",
            "UNKNOWN"
        )

        event_counts[event_type] = (
            event_counts.get(event_type, 0) + 1
        )

    # --------------------------------------------------
    # TRACK / ENTITY INFORMATION
    # --------------------------------------------------

    entities = []

    if not features_df.empty:

        for _, row in features_df.iterrows():

            entity = {}

            for column in features_df.columns:
                entity[column] = safe_value(row[column])

            entity["behavior"] = "NORMAL"
            entity["events"] = []

            entity_id = safe_value(
                row["id"]
            ) if "id" in row else None

            # Find events belonging to this entity
            for event in event_list:

                event_id = event.get("id")

                try:
                    same_id = (
                        int(event_id)
                        == int(entity_id)
                    )
                except Exception:
                    same_id = False

                if same_id:

                    entity["behavior"] = (
                        "UNUSUAL"
                        if event.get("severity") in {
                            "high",
                            "medium"
                        }
                        else event.get(
                            "status",
                            "REVIEW"
                        )
                    )

                    entity["events"].append(
                        event
                    )

            entities.append(entity)

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    result = {

        "project": {
            "name": "Traffic Behavior Understanding System",
            "scenario": "Fixed-camera traffic analysis"
        },

        "summary": {

            "unique_entities": unique_entities,

            "vehicles": vehicles,

            "people": people,

            "total_events": len(event_list),

            "event_counts": event_counts
        },

        "events": event_list,

        "entities": entities,

        "scene": scene_data
    }

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True
    )

    with open(
        args.out,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n[PASS] Analysis result created")

    print(
        f"Entities : {unique_entities}"
    )

    print(
        f"Vehicles : {vehicles}"
    )

    print(
        f"People   : {people}"
    )

    print(
        f"Events   : {len(event_list)}"
    )

    print(
        f"Output   : {args.out}"
    )


if __name__ == "__main__":
    main()