import yaml
import supervision as sv
import numpy as np
import cv2
from collections import deque

class VehicleTracker:
    def __init__(self, config_path="tracking/config.yaml"):
        """
        Initialize the Vehicle Tracker using ByteTrack.
        :param config_path: Path to the tracker configuration YAML file.
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.track_thresh = self.config.get("track_thresh", 0.25)
        self.match_thresh = self.config.get("match_thresh", 0.8)
        self.max_age = self.config.get("max_age", 30)
        self.history_length = self.config.get("history_length", 15)
        
        # Initialize ByteTrack using supervision library (updated arguments for v0.28.0+)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            minimum_matching_threshold=self.match_thresh,
            lost_track_buffer=self.max_age
        )
        
        # Maintain track history mapped by track_id
        # Value is a deque of (x_center, y_center, timestamp)
        self.track_history = {}
        
        # Temporary map to reconstruct string class names returned by tracker
        self.class_name_map = {}
        
    def update(self, detections_list, timestamp):
        """
        Assign persistent IDs to vehicles and maintain recent position history.
        :param detections_list: Output from VehicleDetector (list of dictionaries)
        :param timestamp: Float, current frame timestamp from CameraFeed
        :return: List of TrackedObject dictionaries
        """
        # Handle case where there are no detections in the frame
        if not detections_list:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
            
        # Convert custom Module 2 detection dicts into supervision's Detections object
        xyxy = []
        confidence = []
        class_id = []
        
        self.class_name_map.clear()
        
        for i, det in enumerate(detections_list):
            xyxy.append(det["bbox"])
            confidence.append(det["confidence"])
            class_id.append(i) # Assign list index as a temporary numeric class ID
            self.class_name_map[i] = det["class"] # Remember the actual string class name
            
        sv_detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )
        
        # Run ByteTrack update step
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        tracked_objects = []
        active_track_ids = set()
        
        if len(tracked_detections) > 0:
            # Process each tracked object
            for i in range(len(tracked_detections)):
                box = tracked_detections.xyxy[i]
                c_id = int(tracked_detections.class_id[i])
                t_id = int(tracked_detections.tracker_id[i])
                
                active_track_ids.add(t_id)
                
                # Reconstruct original string class name
                cls_name = self.class_name_map.get(c_id, "unknown")
                
                # Calculate center point
                x1, y1, x2, y2 = box
                x_center = float((x1 + x2) / 2)
                y_center = float((y1 + y2) / 2)
                
                # Initialize deque for new tracks
                if t_id not in self.track_history:
                    self.track_history[t_id] = deque(maxlen=self.history_length)
                    
                # Append current position to history
                self.track_history[t_id].append((x_center, y_center, timestamp))
                
                # Build final output contract format
                tracked_objects.append({
                    "track_id": t_id,
                    "class": cls_name,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "history": list(self.track_history[t_id]) # Oldest to newest due to deque append
                })
                
        # Cleanup completely stale tracks to free memory
        # If a track hasn't been seen in >2.0 seconds, we purge its history completely.
        # Note: ByteTrack internally handles short-term occlusions (up to max_age frames).
        tracks_to_delete = []
        for tid, hist in self.track_history.items():
            last_timestamp = hist[-1][2]
            if timestamp - last_timestamp > 2.0:
                tracks_to_delete.append(tid)
                
        for tid in tracks_to_delete:
            del self.track_history[tid]
            
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
            
            # Base color based on track ID (simple hash)
            color = ((track_id * 37) % 255, (track_id * 73) % 255, (track_id * 149) % 255)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{track_id} {cls_name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw motion trail using historical points
            if len(history) > 1:
                # Extract x,y ignoring timestamp
                pts = np.array([(int(x), int(y)) for x, y, _ in history], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)
                
        return annotated
