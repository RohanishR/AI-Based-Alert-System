import sys
import os
import cv2
import argparse
import time

# Allow importing from root and adjacent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_feed import CameraFeed
from detection.detector import VehicleDetector
from tracker import VehicleTracker

def main():
    parser = argparse.ArgumentParser(description="Test Module 3: Vehicle Tracker Pipeline")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()
    
    source = args.source
    if source.isdigit():
        source = int(source)
        
    print("1. Initializing CameraFeed (Module 1)...")
    feed = CameraFeed(source=source, frame_skip=args.skip)
    
    print("2. Initializing VehicleDetector (Module 2)...")
    detector = VehicleDetector(config_path="detection/config.yaml")
    
    print("3. Initializing VehicleTracker (Module 3)...")
    tracker = VehicleTracker(config_path="tracking/config.yaml")
    
    try:
        feed.open()
        print("\nFull Pipeline active. Press 'Q' to stop the video display.")
        
        while feed.is_opened():
            # Step 1: Read frame
            frame_data = feed.read()
            if frame_data is None:
                print("End of video stream.")
                break
                
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            
            # Step 2: Detect vehicles
            detections, inference_time = detector.detect(frame_data)
            
            # Step 3: Track objects across frames
            track_start = time.time()
            tracked_objects = tracker.update(detections, timestamp)
            track_time = time.time() - track_start
            
            # Step 4: Visualize Results
            annotated_frame = tracker.draw_tracks(frame, tracked_objects)
            
            # Overlay performance metrics
            overlay_text = f"Frame: {frame_data['frame_index']} | Det: {inference_time*1000:.1f}ms | Track: {track_time*1000:.1f}ms"
            cv2.rectangle(annotated_frame, (5, 5), (550, 40), (0, 0, 0), -1)
            cv2.putText(annotated_frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Module 1 + 2 + 3 Tracker Demo", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                print("User requested quit.")
                break
                
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
    finally:
        feed.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Pipeline shut down safely.")

if __name__ == "__main__":
    main()
