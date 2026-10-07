🚦 Traffic Behavior Understanding System

An AI-assisted traffic analysis system that understands vehicle and pedestrian behavior from fixed-camera traffic videos. The system detects and tracks road users, learns normal traffic patterns from the video, identifies unusual behavior, detects possible collisions, and presents the results through an interactive Streamlit dashboard.

🎯 Problem Statement

Traditional traffic monitoring systems mainly focus on detecting and counting vehicles. They do not always explain what the vehicles are doing, whether the behavior is unusual, or why an event may require attention.

This project addresses that problem by building a traffic behavior-understanding pipeline that converts a traffic video into meaningful events such as:

🚗 Unusually stopped vehicles
🛑 Sudden stops
🚫 Wrong-way movement
⚡ Unusually high speed
🚶 Pedestrians on the road
⚠️ Possible collisions

The system provides event information in terms of:

WHO → WHEN → WHAT HAPPENED → WHY IT WAS FLAGGED

💡 Solution Overview


The system accepts a fixed-camera traffic video as input and processes it through multiple stages:

```mermaid
flowchart TD
    A[Traffic Video] --> B[Object Detection<br/>YOLO11s]
    B --> C[Multi-Object Tracking<br/>ByteTrack]
    C --> D[Feature Extraction<br/>Speed • Heading • Position • Stops]
    D --> E[Scene Analysis<br/>Traffic Baselines • Flow • Stop Patterns]
    E --> F[Behavior Reasoning<br/>Rule-Based Analysis]
    F --> G[Collision Reasoning<br/>Contact • Motion • Speed • Stopping]
    G --> H[Event Report<br/>WHO • WHEN • WHY]
    H --> I[Streamlit Dashboard]
🧠 Core Model and Reasoning
1. Object Detection

The system uses YOLO11s for real-time object detection.

The primary classes used for traffic analysis are:

Car
Truck
Bus
Motorcycle
Person

For every detected object, the system obtains:

Bounding box
Object class
Detection confidence
Object center position
2. Multi-Object Tracking

The detected objects are tracked across video frames using ByteTrack.

Each tracked object receives a unique ID.

For example:

Vehicle ID 2
Vehicle ID 4
Vehicle ID 6
Person ID 8

This allows the system to reason about the behavior of the same object over time instead of analyzing individual frames independently.

📊 Data Pipeline

The system processes the input video through the following pipeline.

Step 1 — Detection and Tracking

detect_track.py

Input:

Traffic video

Output:

outputs/tracks.csv
outputs/annotated.mp4

The tracking CSV contains information such as:

frame
time_s
id
class
confidence
bounding box
center position
Step 2 — Feature Extraction

features.py

The tracking information is converted into behavioral features.

The system calculates:

Position
Movement
Relative speed
Heading/direction
Stationary periods
Track duration

Output:

outputs/features_frames.csv
outputs/features_summary.csv
Step 3 — Scene Analysis

scene.py

Instead of relying only on fixed thresholds, the system learns characteristics from the current video.

It estimates:

Dominant traffic flow
Speed baselines
Typical stopping behavior
Local traffic flow
Road-cell statistics

Output:

outputs/scene_stats.json
outputs/scene_overview.png

This allows the system to compare an individual vehicle against the behavior of the surrounding traffic.

Step 4 — Behavior Analysis

rules.py

The extracted features and scene information are used to identify unusual traffic behavior.

Current behavior rules include:

STOPPED_VEHICLE
SUDDEN_STOP
WRONG_WAY
SPEEDING
PEDESTRIAN_ON_ROAD

Output:

outputs/events.csv
outputs/track_status.csv
Step 5 — Possible Collision Detection

collision.py

Collision detection is performed using the tracked movement of vehicles rather than treating object overlap alone as an accident.

The system looks for combinations of evidence such as:

Spatial overlap/proximity
Reduction in vehicle speed
Sudden heading change
Vehicles becoming stationary after contact
Movement of surrounding vehicles

The system deliberately reports:

POSSIBLE_COLLISION

rather than claiming that an accident has been confirmed.

This reduces false claims when the visual evidence is insufficient.

Step 6 — Final Analysis Report

report.py

The intermediate outputs are combined into a single structured result:

outputs/analysis_result.json

The result contains:

Overall statistics
Detected entities
Detected events
Event severity
Event status
Event timing
Scene information
Step 7 — Event Visualization

annotate_events.py

The detected events are overlaid on the original traffic video.

The resulting video is used as visual evidence of the system's analysis.

Step 8 — Web Dashboard

app.py

The final system is presented through a Streamlit web application.

The user can:

Upload a traffic video
Start analysis
Wait for the complete pipeline to execute
View traffic statistics
View detected events
Understand why each event was flagged
View the annotated traffic video
🔎 Evidence & Explanation

The system provides evidence for every detected event.

For example:

Possible collision between Vehicle 4 and Vehicle 6

Time:
0.5s – 3.5s

Evidence:
- Vehicle regions overlapped
- Vehicle 6 reduced speed
- Vehicle 6 changed heading significantly
- Both vehicles became stationary afterward
- Other vehicles continued moving

The dashboard also reports:

Vehicle IDs
Event timestamps
Severity
Event status
Relative speed changes
Stopping duration
Bounding-box overlap
Changes in direction
Surrounding traffic behavior

This allows the result to be reviewed rather than treated as an unexplained AI prediction.

📌 Sample Input and Output
Sample Input

A fixed-camera traffic video containing multiple vehicles and a pedestrian.

Example:

sample_input.mp4
Sample Output

For one demonstrated run, the system detected:

Entities: 11
Vehicles: 10
People: 1
Events: 6

Detected events included:

Stopped vehicle — Vehicle 4
Stopped vehicle — Vehicle 6
Possible collision — Vehicle 4 / Vehicle 6
Possible collision — Vehicle 2
Sudden stop — Vehicle 2
Possible collision — Vehicle 2

Example interpretation:

A possible collision involving two tracked vehicles was detected around 0.5–3.5 seconds. The system observed spatial overlap, a reduction in movement, a major heading change, and both vehicles becoming stationary afterward while surrounding traffic continued moving. The event was therefore flagged for review as a possible collision.

⚠️ Severity and Review Status

The system does not treat every detected event as equally certain.

HIGH

Strong evidence of an unusual or potentially dangerous event.

MEDIUM

Clearly unusual behavior requiring attention.

LOW

Suspicious behavior where the available evidence is insufficient for a strong conclusion.

UNUSUAL

The system considers the behavior significantly different from the learned traffic pattern.

REVIEW

The event should be manually verified.

This approach helps prevent overconfident claims from noisy visual evidence.

🛠️ Technologies Used
Programming
Python 3.13
Computer Vision
OpenCV
Ultralytics YOLO11s
Object Tracking
ByteTrack
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Web Application
Streamlit
Video Processing
FFmpeg
OpenCV VideoWriter
