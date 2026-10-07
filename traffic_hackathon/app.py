import streamlit as st
import os
import sys
import subprocess
import json


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Traffic Behavior Understanding",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🚦 Traffic Behavior Understanding System")

st.write(
    "Upload a traffic video to identify vehicles, understand "
    "their behavior, and detect unusual traffic events."
)

st.divider()


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "mov", "avi", "mkv"]
)


if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)

    video_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"Video uploaded successfully: {uploaded_file.name}"
    )

    # Show uploaded video
    st.video(video_path)

    st.divider()


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        "🔍 Analyze Video",
        type="primary",
        use_container_width=True
    ):

        # -------------------------------------------------
        # RUN COMPLETE PIPELINE
        # -------------------------------------------------

        with st.status(
            "Analyzing traffic video...",
            expanded=True
        ) as status:

            # ---------------------------------------------
            # STEP 1
            # ---------------------------------------------

            st.write(
                "🔍 Detecting and tracking traffic objects..."
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "analyze.py",
                    video_path
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:

                status.update(
                    label="Traffic analysis failed",
                    state="error"
                )

                st.error(result.stderr)

                st.stop()


            # ---------------------------------------------
            # STEP 2
            # ---------------------------------------------

            st.write(
                "📊 Building traffic behavior report..."
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "report.py"
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:

                status.update(
                    label="Report generation failed",
                    state="error"
                )

                st.error(result.stderr)

                st.stop()


            # ---------------------------------------------
            # STEP 3
            # ---------------------------------------------

            st.write(
                "🎥 Creating annotated traffic video..."
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "annotate_events.py",
                    "--video",
                    video_path
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:

                status.update(
                    label="Annotated video generation failed",
                    state="error"
                )

                st.error(result.stderr)

                st.stop()


            # ---------------------------------------------
            # ANALYSIS COMPLETE
            # ---------------------------------------------

            status.update(
                label="Traffic analysis completed!",
                state="complete"
            )


        # =================================================
        # LOAD ANALYSIS RESULT
        # =================================================

        result_file = "outputs/analysis_result.json"


        if not os.path.exists(result_file):

            st.error(
                "Analysis finished, but the final analysis "
                "result was not found."
            )

            st.stop()


        with open(
            result_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        # =================================================
        # DASHBOARD
        # =================================================

        st.header("📊 Traffic Analysis Dashboard")


        summary = data.get(
            "summary",
            {}
        )


        # -------------------------------------------------
        # SUMMARY CARDS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Entities",
                summary.get(
                    "unique_entities",
                    0
                )
            )


        with col2:

            st.metric(
                "Vehicles",
                summary.get(
                    "vehicles",
                    0
                )
            )


        with col3:

            st.metric(
                "People",
                summary.get(
                    "people",
                    0
                )
            )


        with col4:

            st.metric(
                "Events",
                summary.get(
                    "total_events",
                    0
                )
            )


        st.divider()


        # =================================================
        # TRAFFIC EVENTS
        # =================================================

        st.subheader("⚠️ Detected Traffic Events")


        events = data.get(
            "events",
            []
        )


        if not events:

            st.success(
                "No unusual traffic behavior was detected "
                "in the analyzed video."
            )


        else:

            readable_names = {

                "POSSIBLE_COLLISION":
                    "Possible collision",

                "STOPPED_VEHICLE":
                    "Stopped vehicle",

                "SUDDEN_STOP":
                    "Sudden stop",

                "WRONG_WAY":
                    "Wrong-way movement",

                "SPEEDING":
                    "Unusually high speed",

                "PEDESTRIAN_ON_ROAD":
                    "Pedestrian on road"
            }


            for event in events:

                event_type = str(
                    event.get(
                        "type",
                        "UNKNOWN"
                    )
                )


                event_id = event.get(
                    "id",
                    "-"
                )


                start = float(
                    event.get(
                        "start_s",
                        0
                    )
                )


                end = float(
                    event.get(
                        "end_s",
                        0
                    )
                )


                severity = str(
                    event.get(
                        "severity",
                        "unknown"
                    )
                )


                status = str(
                    event.get(
                        "status",
                        "review"
                    )
                )


                reason = str(
                    event.get(
                        "reason",
                        ""
                    )
                )


                readable_type = readable_names.get(
                    event_type,
                    event_type.replace(
                        "_",
                        " "
                    ).title()
                )


                # -----------------------------------------
                # HUMAN-READABLE DESCRIPTION
                # -----------------------------------------

                if event_type == "POSSIBLE_COLLISION":

                    description = (
                        f"A possible collision involving "
                        f"vehicle ID {event_id} was detected "
                        f"between {start:.1f} and {end:.1f} "
                        f"seconds. {reason}"
                    )

                else:

                    description = (
                        f"{readable_type} involving "
                        f"vehicle ID {event_id} was detected "
                        f"between {start:.1f} and "
                        f"{end:.1f} seconds. "
                        f"{reason}"
                    )


                # -----------------------------------------
                # EVENT CARD
                # -----------------------------------------

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### ⚠️ {readable_type}"
                    )

                    st.write(
                        description
                    )

                    st.caption(
                        f"Time: {start:.1f}s – {end:.1f}s   |   "
                        f"Severity: {severity.upper()}   |   "
                        f"Status: {status.upper()}"
                    )


        # =================================================
        # ANNOTATED VIDEO
        # =================================================

        st.divider()

        st.subheader(
            "🎥 Annotated Traffic Video"
        )


        annotated_video = (
            "outputs/final_annotated_h264.mp4"
        )


        if os.path.exists(
            annotated_video
        ):

            st.video(
                annotated_video
            )

        else:

            st.warning(
                "Annotated video was not created."
            )