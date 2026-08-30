import yaml
import supervision as sv
import numpy as np
import cv2
from collections import deque

# Fixed string-to-int mapping for class names used as stable class_ids.
# This ensures ByteTrack always sees the same integer for "car" regardless
# of detection order, which was the root cause of the flickering labels.
CLASS_NAME_TO_ID = {
    "car": 0,
    "motorcycle": 1,
    "auto_rickshaw": 2,
    "bus": 3,
    "truck": 4,
    "bicycle": 5,
    "pedestrian": 6,
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


class VehicleTracker:
    def __init__(self, config_path="tracking/config.yaml"):
        """
        Initialize the Vehicle Tracker using ByteTrack.
        :param config_path: Path to the tracker configuration YAML file.
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.track_thresh = self.config.get("track_thresh", 0.3)
        self.match_thresh = self.config.get("match_thresh", 0.85)
        self.max_age = self.config.get("max_age", 50)
        self.history_length = self.config.get("history_length", 15)
        self.min_consecutive = self.config.get("minimum_consecutive_frames", 3)
        
        # Initialize ByteTrack (supervision v0.28.0+ API)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            minimum_matching_threshold=self.match_thresh,
            lost_track_buffer=self.max_age,
            minimum_consecutive_frames=self.min_consecutive
        )
        
        # Maintain track history: track_id -> deque of (x_center, y_center, timestamp)
        self.track_history = {}
        
        # Maintain the last known class per track_id to prevent class-label flickering
        self.track_class = {}
        
    def update(self, detections_list, timestamp):
        """
        Assign persistent IDs to vehicles and maintain recent position history.

        :param detections_list: Output from VehicleDetector (list of dicts with
               keys "class", "confidence", "bbox")
        :param timestamp: Float, current frame timestamp from CameraFeed
        :return: List of TrackedObject dictionaries following the interface contract:
                 {
                     "track_id": int,
                     "class": str,
                     "bbox": (x1, y1, x2, y2),
                     "history": [(x_center, y_center, timestamp), ...]  # oldest first
                 }
        """
        # Handle empty frame
        if not detections_list:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
            
        # Convert Module 2 detections into supervision Detections object
        # Using a STABLE class_id mapping (not list index!) to prevent label flickering
        xyxy = []
        confidence = []
        class_id = []
        
        for det in detections_list:
            xyxy.append(det["bbox"])
            confidence.append(det["confidence"])
            # Map string class name to a fixed integer
            cls_int = CLASS_NAME_TO_ID.get(det["class"], -1)
            if cls_int == -1:
                continue  # Skip unknown classes
            class_id.append(cls_int)
            
        if not xyxy:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
            
        sv_detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int)
        )
        
        # Run ByteTrack update step
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        tracked_objects = []
        
        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                box = tracked_detections.xyxy[i]
                c_id = int(tracked_detections.class_id[i])
                t_id = int(tracked_detections.tracker_id[i])
                
                # Resolve class name from stable mapping
                cls_name = CLASS_ID_TO_NAME.get(c_id, "unknown")
                
                # Sticky class label: once a track is assigned a class, it keeps it.
                # This prevents a car flickering to "truck" and back across frames.
                if t_id in self.track_class:
                    cls_name = self.track_class[t_id]
                else:
                    self.track_class[t_id] = cls_name
                
                # Calculate center point
                x1, y1, x2, y2 = box
                x_center = float((x1 + x2) / 2)
                y_center = float((y1 + y2) / 2)
                
                # Initialize history deque for new tracks
                if t_id not in self.track_history:
                    self.track_history[t_id] = deque(maxlen=self.history_length)
                    
                # Append current position
                self.track_history[t_id].append((x_center, y_center, timestamp))
                
                # Build output
                tracked_objects.append({
                    "track_id": t_id,
                    "class": cls_name,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "history": list(self.track_history[t_id])
                })
                
        # Cleanup stale tracks (not seen for > 3 seconds)
        stale_ids = [
            tid for tid, hist in self.track_history.items()
            if timestamp - hist[-1][2] > 3.0
        ]
        for tid in stale_ids:
            del self.track_history[tid]
            self.track_class.pop(tid, None)
            
        return tracked_objects
        
    def draw_tracks(self, frame, tracked_objects):
        """
        Draw bounding boxes, track IDs, vehicle classes, and motion trails.
        :param frame: Image array
        :param tracked_objects: Output from update() method
        :return: Annotated image array
        """
        annotated = frame.copy()
        
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            track_id = obj["track_id"]
            cls_name = obj["class"]
            history = obj["history"]
            
            # Stable color per track ID using golden-ratio hashing for better spread
            hue = int((track_id * 137.508) % 180)
            color_bgr = cv2.cvtColor(
                np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR
            )[0][0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"ID:{track_id} {cls_name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            # Draw motion trail with fading effect
            if len(history) > 1:
                pts = [(int(x), int(y)) for x, y, _ in history]
                for j in range(1, len(pts)):
                    # Fade: older segments are thinner
                    thickness = max(1, int(2 * j / len(pts)) + 1)
                    cv2.line(annotated, pts[j - 1], pts[j], color, thickness)
                    
        return annotated
