import time
import cv2
import yaml
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
            
        self.conf_threshold = self.config.get("confidence_threshold", 0.4)
        self.model_path = self.config.get("model_path", "yolov8n.pt")
        self.target_classes = set(self.config.get("target_classes", []))
        self.class_mapping = self.config.get("class_mapping", {})
        
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
        
        # Run inference (verbose=False avoids console spam per frame)
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        inference_time = time.time() - start_time
        
        detections = []
        
        if len(results) > 0:
            result = results[0]  # Get results for the single image
            boxes = result.boxes
            
            for box in boxes:
                # Extract properties
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Check mapped name in config first
                class_name = self.class_mapping.get(cls_id, None)
                
                # Fallback to model's default names if missing from config
                if not class_name and hasattr(self.model, 'names') and cls_id in self.model.names:
                    raw_name = self.model.names[cls_id]
                    class_name = "pedestrian" if raw_name == "person" else raw_name
                
                if not class_name:
                    class_name = f"class_{cls_id}"
                
                # Filter strictly to target classes
                if class_name in self.target_classes:
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": (float(x1), float(y1), float(x2), float(y2))
                    })
                    
        return detections, inference_time
        
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
