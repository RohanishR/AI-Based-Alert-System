import time
import cv2
import yaml
import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, config_path="detection/config.yaml"):
        """
        Initialize the Vehicle Detector.
        :param config_path: Path to the detector configuration YAML file.
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.conf_threshold = self.config.get("confidence_threshold", 0.25)
        self.model_path = self.config.get("model_path", "yolov8n.pt")
        self.input_size = self.config.get("input_size", 640)
        self.target_classes = set(self.config.get("target_classes", []))
        self.class_mapping = self.config.get("class_mapping", {})
        
        # Build a reverse lookup: COCO class_id -> our standardized name
        # This is the ONLY source of truth for class name resolution
        self._id_to_name = {}
        for cid, name in self.class_mapping.items():
            self._id_to_name[int(cid)] = name
        
        # Initialize YOLOv8 model
        print(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully.")
        
    def detect(self, frame_data):
        """
        Run object detection on the provided frame.
        :param frame_data: dict containing "frame", "frame_index", "timestamp" from Module 1
        :return: Tuple containing:
                 - detections: list of dictionaries mapping exactly to the spec format
                 - inference_time: time taken in seconds for the forward pass
        """
        frame = frame_data["frame"]
        
        start_time = time.time()
        
        # Run inference with fixed input size for consistent results
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self.input_size
        )
        
        inference_time = time.time() - start_time
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                # Batch extraction for speed (avoids per-box Python loop overhead)
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)
                xyxys = boxes.xyxy.cpu().numpy().astype(float)
                
                for idx in range(len(cls_ids)):
                    cls_id = int(cls_ids[idx])
                    conf = float(confs[idx])
                    x1, y1, x2, y2 = xyxys[idx]
                    
                    # Resolve class name via our mapping
                    class_name = self._id_to_name.get(cls_id, None)
                    
                    # Fallback: use model's native name for unmapped IDs
                    if class_name is None and hasattr(self.model, 'names') and cls_id in self.model.names:
                        raw_name = self.model.names[cls_id]
                        class_name = "pedestrian" if raw_name == "person" else raw_name
                    
                    if class_name is None:
                        continue  # Skip completely unknown classes
                    
                    # Only keep target classes
                    if class_name not in self.target_classes:
                        continue
                        
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": (float(x1), float(y1), float(x2), float(y2))
                    })
        
        # Suppress pedestrian detections that overlap with motorcycles/bicycles
        # (a rider on a bike is not a pedestrian)
        detections = self._suppress_riders(detections)
                    
        return detections, inference_time
    
    @staticmethod
    def _box_iou(box_a, box_b):
        """Compute IoU between two (x1,y1,x2,y2) boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0
    
    @staticmethod
    def _suppress_riders(detections, iou_threshold=0.3):
        """
        Remove pedestrian detections that overlap with motorcycle/bicycle boxes.
        In Indian traffic footage, YOLO often detects both the two-wheeler AND the 
        rider as separate objects. The rider should not be counted as a pedestrian.
        """
        two_wheelers = [d for d in detections if d["class"] in ("motorcycle", "bicycle")]
        if not two_wheelers:
            return detections
            
        filtered = []
        for det in detections:
            if det["class"] == "pedestrian":
                # Check if this pedestrian overlaps significantly with any two-wheeler
                overlaps = any(
                    VehicleDetector._box_iou(det["bbox"], tw["bbox"]) > iou_threshold
                    for tw in two_wheelers
                )
                if overlaps:
                    continue  # Skip this pedestrian — it's a rider
            filtered.append(det)
        return filtered
        
    def draw_detections(self, frame, detections, inference_time=None):
        """
        Draw bounding boxes and labels on a frame for visualization.
        :param frame: Original image array
        :param detections: List of processed detection dictionaries
        :param inference_time: Optional processing time to display FPS
        :return: Annotated image array
        """
        # Create a copy so we don't modify the original reference
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            conf = det["confidence"]
            
            # Draw bounding box (Green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw label text (Black)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        # Draw FPS overlay if timing is provided
        if inference_time and inference_time > 0:
            fps = 1.0 / inference_time
            cv2.putText(annotated, f"Inference: {fps:.1f} FPS", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
        return annotated
