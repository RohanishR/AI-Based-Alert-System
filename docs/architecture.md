# Architecture Document
## Intersection Collision Prediction using Fixed Roadside Cameras (V2I-Based Alert System)

---

## 1. Overview

This system uses a fixed roadside camera to detect vehicles at an intersection, track their movement over time, predict their future trajectories, calculate collision risk between vehicle pairs, and simulate a Vehicle-to-Infrastructure (V2I) safety alert. It is designed as a software-only complement to India's proposed AIS-230 Vehicle-to-Vehicle (V2V) communication mandate, addressing the infrastructure side of intersection safety using computer vision instead of in-vehicle hardware.

### 1.1 Goals
- Detect and classify vehicles (car, bike, auto, bus) in real time from a single fixed camera
- Maintain a persistent identity for each vehicle as it moves through the frame
- Predict short-horizon (2–5 second) future positions for each vehicle
- Quantify collision risk between vehicle pairs using Time-to-Collision (TTC)
- Simulate a V2I-style alert broadcast when risk crosses a defined threshold
- Evaluate the pipeline end-to-end against a manually labeled test set

### 1.2 Non-Goals (Explicit Scope Boundaries)
- Not building or interfacing with real AIS-230 / C-V2X radio hardware
- Not predicting non-linear evasive maneuvers (sudden swerves, emergency braking response)
- Not handling multi-camera fusion across intersections (single fixed camera only)
- Not deploying to real-time embedded hardware (target is a laptop/server-class demo)

### 1.3 Design Principles
- **Modularity**: Each of the six stages is an independent module with a clearly defined input/output contract, so modules can be developed, tested, and swapped independently (e.g., replacing ByteTrack with DeepSORT without touching the prediction module).
- **Pretrained-first**: Wherever possible, use pretrained/off-the-shelf models (YOLOv8, ByteTrack) rather than training from scratch, to keep the project feasible within a semester.
- **Fail-soft**: A failure in one module (e.g., a missed detection) should degrade gracefully — a vehicle drops out of tracking rather than crashing the pipeline.
- **Explainability over complexity**: Where a simple, interpretable method (Kalman filter, TTC formula) is adequate, it is preferred over a more complex deep-learning alternative, since the system needs to be explainable in an academic evaluation setting.

---

## 2. High-Level System Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Camera   │────▶│  2. Vehicle  │────▶│  3. Multi-   │────▶│  4. Trajectory│────▶│  5. Risk     │────▶│  6. Alert    │
│     Feed     │     │  Detection   │     │  Object      │     │  Prediction   │     │  Scoring     │     │  Simulation  │
│  (Input)     │     │  (YOLOv8)    │     │  Tracking    │     │  (Kalman      │     │  (TTC)       │     │  (Dashboard) │
│              │     │              │     │  (ByteTrack) │     │  Filter)      │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.1 Per-Frame Sequence Diagram

```
Frame N arrives
   │
   ▼
[Detection]  YOLOv8 inference ──► list of (class, conf, bbox)
   │
   ▼
[Tracking]   ByteTrack.update(detections) ──► list of (track_id, bbox)
   │
   ▼
[Prediction] For each track_id: Kalman.predict() + Kalman.update(new_pos)
             ──► predicted (x, y) at t+1s, t+2s, t+3s
   │
   ▼
[Risk Scoring] For each pair (track_id_i, track_id_j) within proximity radius:
               compute TTC(i, j)
   │
   ▼
[Alert]      If TTC < threshold AND pair not already alerted in last N seconds:
             generate_alert() ──► push to dashboard queue
   │
   ▼
Frame N+1 (repeat)
```

### 2.2 Processing Mode
- **Offline/batch mode** (primary, used for evaluation): process a recorded video file frame-by-frame, log all detections/tracks/alerts to disk for later metric computation.
- **Live/near-real-time demo mode** (used for presentation): process frames from a webcam or recorded video played in real time, rendering bounding boxes, trails, and alerts on-screen as it runs, plus mirroring alerts to the Streamlit dashboard.

---

## 3. Module Breakdown

### Module 1 — Camera Feed (Input Layer)

- **Purpose**: Ingests raw video, frame by frame, from a fixed intersection camera (dataset video or live/recorded footage).
- **Input**: `.mp4` / `.avi` video file, or camera stream (webcam index / RTSP URL).
- **Output**: Sequence of individual frames (images) passed downstream, each tagged with a frame index and timestamp.
- **Tech**: OpenCV (`cv2.VideoCapture`).
- **Configurable parameters**:
  - `frame_skip` — process every Nth frame to reduce compute load (e.g., process every 2nd frame at 30fps → effective 15fps pipeline)
  - `resize_dim` — downscale resolution before detection to speed up inference (e.g., 1280×720 → 640×384)
- **Responsibilities**:
  - Normalize frame rate/resolution before passing downstream
  - Attach a monotonically increasing frame timestamp (used later for velocity and TTC calculations)
  - Handle end-of-stream / dropped-frame conditions gracefully

### Module 2 — Vehicle Detection

- **Purpose**: Identifies and classifies every vehicle present in each frame.
- **Input**: Single video frame (image array, resized per config).
- **Output**: List of detections per frame — `[class, confidence, bounding_box(x1,y1,x2,y2)]` for each vehicle (car, bike, auto, bus).
- **Tech**: YOLOv8 (Ultralytics), pretrained on COCO, fine-tuned on an Indian traffic dataset (IDD / UA-DETRAC / Roboflow Indian Vehicle Dataset) to add auto-rickshaw and two-wheeler classes not well represented in COCO.
- **Model variant choice**: YOLOv8n (nano) recommended for this project — smaller, faster, sufficient accuracy for a fixed-camera, moderate-resolution use case; avoids needing high-end GPU access for training/inference.
- **Fine-tuning approach**:
  - Start from COCO-pretrained YOLOv8n weights
  - Freeze backbone layers initially, fine-tune detection head on the Indian dataset for faster convergence
  - Classes: `car`, `motorcycle`, `auto_rickshaw`, `bus`, `truck`, `bicycle`, `pedestrian` (pedestrian included for future extension to vehicle-pedestrian risk, even if not scored in the current version)
- **Confidence threshold**: Detections below `conf_threshold = 0.4` (tunable) are discarded before being passed to the tracker, to reduce false positives feeding into tracking.
- **Owner**: Member A

### Module 3 — Multi-Object Tracking

- **Purpose**: Links detections across frames, assigning each vehicle a persistent ID so its motion over time can be studied.
- **Input**: Per-frame detections from Module 2 (post confidence-thresholding).
- **Output**: Tracked objects with `[track_id, class, bounding_box, position_history]`.
- **Tech**: ByteTrack (via the `supervision` library) — chosen over DeepSORT for this project because:
  - No separate appearance/Re-ID CNN needs to be trained or loaded (simpler pipeline, lower compute)
  - Associates *every* detection box, including low-confidence ones, which helps recover vehicles that are briefly occluded (common at Indian intersections with high vehicle density)
  - Well-documented 3-line integration via `supervision`
- **Internal mechanism**:
  1. Predict each existing track's expected position this frame (motion model)
  2. Compute IoU (Intersection-over-Union) between predicted boxes and new detections
  3. Solve optimal assignment via the Hungarian algorithm
  4. Matched → keep ID; unmatched detection → new ID; unmatched track → mark "lost," delete after `max_age` frames without a match
- **Configurable parameters**:
  - `track_thresh` — confidence threshold for a detection to start a new track
  - `match_thresh` — IoU threshold for matching
  - `max_age` — number of frames a lost track is kept alive before deletion (higher = more tolerant of brief occlusion, but risks incorrect re-matching)
- **Owner**: Member A

> **Integration boundary**: Module 3's output (tracked vehicle IDs + position history, typically the last 10–15 frames per `track_id`) is the input contract for Module 4. This is the handoff point between Member A's and Member B's work — agreed upon early so both can develop independently against this contract.

### Module 4 — Trajectory Prediction

- **Purpose**: Projects each tracked vehicle's future position a few seconds ahead, based on its recent motion.
- **Input**: Position history per `track_id` from Module 3 (list of `(x, y, timestamp)` tuples).
- **Output**: Predicted future positions `(x, y)` at future timestamps (e.g., t+1s, t+2s, t+3s), per vehicle.
- **Tech**: Kalman filter (`filterpy` library) — a classical statistical estimation method, not deep learning.
- **State representation**: `state = [x, y, vx, vy]` (position and velocity in image-plane or calibrated ground-plane coordinates)
- **Predict/Update cycle**:
  - `predict()`: projects the state forward using a constant-velocity motion model
  - `update(measurement)`: corrects the prediction using the newly observed position from tracking, weighted by the filter's uncertainty (covariance)
- **Coordinate calibration note**: Raw pixel coordinates from the camera do not correspond linearly to real-world distances/speed, especially at an angle. For meaningful TTC values (in real seconds/meters), a simple homography transform (mapping known reference points in the frame — e.g., lane markings of known width — to real-world coordinates) should be applied before feeding positions into the Kalman filter. This calibration step is a one-time setup per camera placement.
- **Prediction horizon**: Multi-step prediction (t+1s to t+3s) is generated by repeatedly calling `predict()` without an intervening `update()`, projecting further into the future each call.
- **Owner**: Member B (developed/tested against simulated placeholder position data before Module 3's real output is available, then integrated at the agreed handoff point)

### Module 5 — Risk Scoring (Time-to-Collision)

- **Purpose**: Evaluates every pair of tracked vehicles to determine if their predicted paths will cross, and how soon.
- **Input**: Predicted trajectories from Module 4, for all currently tracked vehicles.
- **Output**: `TTC` (Time-to-Collision, in seconds) per at-risk vehicle pair; `None`/no output if no collision course.
- **Tech**: Physics-based calculation (relative position, relative velocity, closing speed) — no ML involved.
- **Core formula** (for two vehicles A and B):
  ```
  relative_position = position_B - position_A
  relative_velocity  = velocity_B - velocity_A
  closing_speed      = -(relative_position · relative_velocity) / |relative_position|

  if closing_speed <= 0:
      no risk (vehicles moving apart or parallel)
  else:
      TTC = |relative_position| / closing_speed
  ```
- **Pairwise scope reduction**: To avoid O(n²) computation blowing up with many vehicles, only compute TTC for vehicle pairs within a configurable `proximity_radius` of each other and both moving toward the intersection zone — vehicles far apart or moving away are skipped early.
- **Threshold**: `TTC_THRESHOLD = 2.0` seconds (tunable) — below this, a pair is flagged as high risk; between `2.0`–`4.0`s can optionally be flagged as "medium" severity.
- **Owner**: Member B

### Module 6 — Alert Simulation

- **Purpose**: When TTC drops below a safety threshold, generates a structured alert representing a simulated V2I broadcast, and displays it.
- **Input**: Flagged vehicle pairs + TTC value from Module 5.
- **Output**: Alert object (JSON-like):
  ```json
  {
    "alert_type": "forward_collision_warning",
    "timestamp": 1699999999.123,
    "vehicles_involved": [7, 12],
    "time_to_collision": 1.4,
    "severity": "high",
    "location": "Intersection_A"
  }
  ```
- **Tech**: Streamlit dashboard for live display; alerts also logged to a local file/DB for later precision/recall evaluation.
- **Deduplication logic**: An `active_alerts` dictionary keyed by `(track_id_a, track_id_b)` (sorted tuple) tracks the last alert time for each vehicle pair; a new alert for the same pair is only re-raised if more than `cooldown_seconds` (e.g., 5s) has passed, to avoid flooding the dashboard with repeated alerts for a single ongoing risk event.
- **Severity mapping**: `TTC < 1.0s → "high"`, `1.0s ≤ TTC < 2.0s → "medium"`.
- **Owner**: Member B

---

## 4. Data Flow Summary

| Stage | Input | Output | Format |
|---|---|---|---|
| Camera Feed | Video file/stream | Frames | Image array (per frame), + timestamp |
| Detection | Frame | Bounding boxes + class | `[class, conf, x1,y1,x2,y2]` |
| Tracking | Detections (per frame) | Tracked IDs + history | `{track_id: [(x,y,t), ...]}` |
| Prediction | Position history | Future positions | `{track_id: predicted (x,y) @ t+1s..t+3s}` |
| Risk Scoring | Predicted positions (pairs) | TTC value | `float (seconds)` or `None` |
| Alert | TTC + vehicle IDs | Alert message | JSON object, displayed on dashboard + logged |

---

## 5. Tech Stack

| Layer | Tool / Library | Notes |
|---|---|---|
| Language | Python 3.10+ | |
| Video I/O | OpenCV | Frame reading, resizing, drawing overlays |
| Detection | YOLOv8n (Ultralytics) | Pretrained + fine-tuned |
| Tracking | ByteTrack (via `supervision`) | Motion-based, no Re-ID CNN needed |
| Trajectory Prediction | `filterpy` (Kalman filter) | Classical method, lightweight |
| Risk Scoring | NumPy | Vector math for TTC |
| Dashboard / Alert Display | Streamlit | Live alert table + optional video overlay |
| Dataset | UA-DETRAC, IDD (India Driving Dataset) | Detection/tracking fine-tuning |
| Annotation (for test set) | CVAT | Manual near-miss labeling for evaluation |
| Evaluation | `pycocotools` (mAP), `motmetrics` (MOTA/ID-switch) | Standard CV evaluation libraries |
| Experiment tracking (optional) | TensorBoard or simple CSV logs | Track training/evaluation runs |

---

## 6. Folder Structure

```
intersection-collision-prediction/
├── data/
│   ├── raw_videos/
│   ├── datasets/               # UA-DETRAC / IDD subsets
│   ├── calibration/             # homography reference points per camera
│   └── labeled_test_set/        # CVAT-annotated near-miss clips
├── detection/
│   ├── train.py
│   ├── infer.py
│   ├── config.yaml               # conf_threshold, classes, model variant
│   └── weights/
├── tracking/
│   ├── tracker.py
│   └── config.yaml               # track_thresh, match_thresh, max_age
├── prediction/
│   ├── kalman_filter.py
│   └── calibration.py            # pixel-to-real-world transform
├── risk_scoring/
│   ├── ttc.py
│   └── config.yaml               # TTC_THRESHOLD, proximity_radius
├── alerts/
│   ├── alert_generator.py
│   ├── dashboard.py              # Streamlit app
│   └── alert_log.jsonl           # persisted alert history
├── evaluation/
│   ├── detection_metrics.py      # mAP
│   ├── tracking_metrics.py       # MOTA, ID-switches
│   └── alert_metrics.py          # precision/recall vs labeled test set
├── pipeline/
│   ├── run_offline.py            # batch mode: video file → logged results
│   └── run_live.py               # live/demo mode: webcam → real-time overlay + dashboard
├── docs/
│   └── architecture.md           # this file
├── requirements.txt
└── README.md
```

---

## 7. Interface Contracts Between Modules

Defining these explicitly lets Member A and Member B develop independently without breaking each other's code.

```python
# Contract: Detection → Tracking
Detection = {
    "class": str,          # "car" | "motorcycle" | "auto_rickshaw" | "bus" | "truck"
    "confidence": float,   # 0.0–1.0
    "bbox": (float, float, float, float)  # x1, y1, x2, y2 in pixel coords
}

# Contract: Tracking → Prediction
TrackedObject = {
    "track_id": int,
    "class": str,
    "bbox": (float, float, float, float),
    "history": [(float, float, float), ...]  # (x_center, y_center, timestamp), most recent last
}

# Contract: Prediction → Risk Scoring
PredictedTrajectory = {
    "track_id": int,
    "predictions": [(float, float, float), ...]  # (x, y, future_timestamp)
    "velocity": (float, float)  # vx, vy at current time
}

# Contract: Risk Scoring → Alert
RiskEvent = {
    "vehicle_pair": (int, int),  # track_id_a, track_id_b
    "ttc": float,
    "severity": str  # "high" | "medium"
}
```

---

## 8. Team Ownership

| Member | Modules | Focus |
|---|---|---|
| **Member A** | 1, 2, 3 | Detection & Tracking |
| **Member B** | 4, 5, 6 | Prediction, Risk Scoring, Alerting |
| **Both** | Dataset labeling, calibration setup, integration testing, evaluation, report | Shared |

---

## 9. Evaluation Metrics

| Component | Metric | Tool |
|---|---|---|
| Detection | mAP@0.5 per class | `pycocotools` |
| Tracking | MOTA, ID-switch count, ID-F1 | `motmetrics` |
| Alert System | Precision, Recall against labeled near-miss test set | Custom script comparing generated alerts vs. CVAT ground truth |
| End-to-end | Qualitative demo review (video walkthrough with overlays) | Manual review |

**Evaluation protocol**: A held-out test clip (not used in fine-tuning) is manually annotated in CVAT with ground-truth near-miss events (timestamp + vehicle pair). The pipeline is run on this clip, and generated alerts are matched against ground truth within a tolerance window (e.g., ±1 second) to compute precision/recall.

---

## 10. Known Limitations

- Trajectory prediction assumes near-linear motion; sudden braking, sharp turns, or erratic lane changes reduce prediction accuracy.
- This is a **simulation** of V2I alerting — no real broadcast to physical vehicles occurs, since that requires regulated hardware and spectrum access outside project scope.
- Detection/tracking accuracy is expected to be lower on Indian mixed-traffic footage than on benchmark datasets, due to non-lane-disciplined traffic and vehicle diversity (autos, two-wheelers).
- Without proper camera calibration (pixel-to-real-world mapping), TTC values are only relatively meaningful, not absolutely accurate in real seconds — calibration quality directly affects result validity.
- Single fixed camera only; occlusion from large vehicles (buses/trucks) blocking smaller ones behind them is a known failure mode.

---

## 11. Future Scope

- Extend risk scoring to vehicle-pedestrian pairs (school zones, informal crossings) — pedestrian class is already included in the detection schema for this reason
- Apply the same pipeline to roundabouts, railway crossings, and toll plazas
- Replace simulated alerts with real V2I broadcast once AIS-230 infrastructure becomes available (post-2028)
- Multi-camera fusion for intersections with occluded viewing angles
- Replace linear Kalman filter with a learned trajectory model (e.g., Bi-LSTM) if time/data permits, as explored in comparable published work
